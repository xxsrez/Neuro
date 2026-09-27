using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using Neuro;

namespace Neuro.Cli;

public sealed record EpochRecord(int Epoch, Metrics Train, Metrics Validation, double TrainSeconds, double EvaluationSeconds);
public sealed record WeightFile(int FormatVersion, Configuration Configuration, int Epoch, double[][] Parameters);
public sealed record PointView(double X, double Y, double TargetX, double TargetY, double PredictedX, double PredictedY, double Error);
public sealed record GradientView(int Block, double ParameterRms, double InputRms);

public static class Training
{
    public static readonly JsonSerializerOptions Json = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = false,
        Converters = { new JsonStringEnumConverter() }
    };
    public static void Save(string path, object value) => File.WriteAllText(path, JsonSerializer.Serialize(value, Json));

    public static void Run(Configuration c)
    {
        if (Directory.Exists(c.Output) || File.Exists(c.Output)) throw new ArgumentException("Output already exists; choose a new --out directory.");
        Directory.CreateDirectory(c.Output);
        Save(Path.Combine(c.Output, "config.json"), c);
        Save(Path.Combine(c.Output, "environment.json"), new
        {
            runtime = RuntimeInformation.FrameworkDescription, os = RuntimeInformation.OSDescription,
            architecture = RuntimeInformation.ProcessArchitecture.ToString(), cpuThreads = Environment.ProcessorCount,
            git = GitRevision(), generatedUtc = DateTimeOffset.UtcNow, motionVersion = Motion.Version,
            randomVersion = "splitmix64-v1", normalization = "divide coordinates by pi",
            initialization = "uniform +/- sqrt(6/(fanIn+fanOut)), zero bias", singleThread = true
        });
        var train = Motion.Generate(c.TrainCount, c.TrainSeed, c.Steps, c.H);
        var validation = Motion.Generate(c.ValidationCount, c.ValidationSeed, c.Steps, c.H);
        var affine = new AffineBaseline(train);
        var identityValidation = Evaluation.Measure(validation, x => x);
        var affineValidation = Evaluation.Measure(validation, affine.Predict);
        var model = new Network(c.Width, c.Blocks, c.Activation, c.Alpha, c.WeightSeed);
        var optimizer = new Optimizer(model.Parameters, c.Optimizer, c.LearningRate);
        var shuffle = new RandomStream(c.ShuffleSeed);
        int[] order = Enumerable.Range(0, train.Length).ToArray();
        var history = new List<EpochRecord>();
        Metrics firstValidation = Evaluation.Measure(validation, model.Forward);
        Metrics firstTrain = Evaluation.Measure(train, model.Forward);
        history.Add(new(0, firstTrain, firstValidation, 0, 0));
        double bestMse = firstValidation.Mse; int bestEpoch = 0; var best = model.Snapshot();
        using var csv = new StreamWriter(Path.Combine(c.Output, "metrics.csv"));
        csv.WriteLine("epoch,train_mse,validation_mse,train_distance,validation_distance,train_seconds,evaluation_seconds");
        WriteRow(csv, history[0]);
        Console.WriteLine($"Parameters={model.ParameterCount}; width={c.Width}; blocks={c.Blocks}; activation={c.Activation}; optimizer={c.Optimizer}");
        Console.WriteLine($"Validation baselines: identity={identityValidation.Mse:G6}, affine={affineValidation.Mse:G6}; initial={firstValidation.Mse:G6}; acceptance<={c.AcceptanceMse}");
        var total = Stopwatch.StartNew();
        for (int epoch = 1; epoch <= c.Epochs; epoch++)
        {
            shuffle.Shuffle(order); var timer = Stopwatch.StartNew();
            for (int start = 0; start < order.Length; start += c.BatchSize)
            {
                int count = Math.Min(c.BatchSize, order.Length - start);
                double batchLoss = Batches.Accumulate(model, train, order, start, count);
                if (!double.IsFinite(batchLoss)) throw new ArithmeticException($"Non-finite batch loss at epoch {epoch}");
                optimizer.Step();
            }
            double trainTime = timer.Elapsed.TotalSeconds; timer.Restart();
            var trainMetric = Evaluation.Measure(train, model.Forward);
            var validationMetric = Evaluation.Measure(validation, model.Forward);
            var record = new EpochRecord(epoch, trainMetric, validationMetric, trainTime, timer.Elapsed.TotalSeconds);
            history.Add(record); WriteRow(csv, record);
            if (validationMetric.Mse < bestMse)
            {
                bestMse = validationMetric.Mse; bestEpoch = epoch; best = model.Snapshot();
            }
            if (epoch <= 3 || epoch % 10 == 0 || epoch == c.Epochs)
                Console.WriteLine($"epoch={epoch} train={trainMetric.Mse:G6} val={validationMetric.Mse:G6} distance={validationMetric.MeanDistance:F4} train_s={trainTime:F3}");
        }
        total.Stop();
        Save(Path.Combine(c.Output, "last-weights.json"), new WeightFile(1, c, c.Epochs, model.Snapshot()));
        Save(Path.Combine(c.Output, "best-weights.json"), new WeightFile(1, c, bestEpoch, best));
        model.Restore(best);
        var bestTrain = Evaluation.Measure(train, model.Forward); var bestValidation = Evaluation.Measure(validation, model.Forward);
        // Test data is generated and evaluated only after validation-based selection.
        var test = Motion.Generate(c.TestCount, c.TestSeed, c.Steps, c.H);
        var testMetric = Evaluation.Measure(test, model.Forward);
        bool accepted = bestValidation.Mse <= c.AcceptanceMse && bestValidation.Mse < firstValidation.Mse &&
            bestValidation.Mse < identityValidation.Mse && bestValidation.Mse < affineValidation.Mse;
        var gradients = Diagnostics(model, validation);
        var grid = Motion.Grid(c.GridSide, c.Steps, c.H);
        var points = new PointView[grid.Length];
        for (int i = 0; i < grid.Length; i++)
        {
            var s = grid[i]; var y = model.Forward(s.Input);
            points[i] = new(s.Input[0] * Math.PI, s.Input[1] * Math.PI, s.Target[0] * Math.PI, s.Target[1] * Math.PI,
                y[0] * Math.PI, y[1] * Math.PI, Math.Sqrt(2 * Loss.Mse(y, s.Target)) * Math.PI);
        }
        const int activationSide = 21;
        var activationGrid = Motion.Grid(activationSide, c.Steps, c.H);
        var activations = new double[c.Blocks][];
        for (int b = 0; b < c.Blocks; b++) activations[b] = new double[activationGrid.Length * c.Width];
        for (int i = 0; i < activationGrid.Length; i++)
        {
            model.Forward(activationGrid[i].Input);
            for (int b = 0; b < c.Blocks; b++) Array.Copy(model.Blocks[b].Activated, 0, activations[b], i * c.Width, c.Width);
        }
        var trajectories = new List<double[][]>();
        foreach (var initial in new[] { (0.4, 0.8), (-1.8, 1.0), (2.0, -0.6), (-2.4, -2.0) })
        {
            var path = new List<double[]> { new double[] { initial.Item1, initial.Item2 } }; var (x, y) = initial;
            for (int s = 0; s < c.Steps; s++) { (x, y) = Motion.Step(x, y, c.H); path.Add([x, y]); }
            trajectories.Add(path.ToArray());
        }
        var summary = new
        {
            parameterCount = model.ParameterCount, bestEpoch, accepted, acceptanceMse = c.AcceptanceMse,
            initialValidation = firstValidation, bestTrain, bestValidation, test = testMetric,
            identityValidation, affineValidation,
            identityTest = Evaluation.Measure(test, x => x), affineTest = Evaluation.Measure(test, affine.Predict),
            affineCoefficients = affine.Coefficients, trainingSeconds = total.Elapsed.TotalSeconds,
            optimizerSteps = optimizer.StepCount, diagnosticsSamples = Math.Min(64, validation.Length),
            checkpointNote = "Weights for evaluation only; optimizer/RNG state not saved, resume unsupported."
        };
        Save(Path.Combine(c.Output, "summary.json"), summary);
        Save(Path.Combine(c.Output, "history.json"), history);
        var report = new { config = c, summary, history, grid = points, activationSide, activations, gradients, trajectories };
        Save(Path.Combine(c.Output, "report-data.json"), report);
        Report.Write(Path.Combine(c.Output, "report.html"), c, history, points, gradients, summary, JsonSerializer.Serialize(report, Json));
        Console.WriteLine($"Best epoch={bestEpoch}; validation={bestValidation.Mse:G6}; test={testMetric.Mse:G6}; accepted={accepted}; seconds={total.Elapsed.TotalSeconds:F2}");
        Console.WriteLine($"Report: {Path.GetFullPath(Path.Combine(c.Output, "report.html"))}");
    }

    public static void EvaluateSaved(string directory)
    {
        var file = JsonSerializer.Deserialize<WeightFile>(File.ReadAllText(Path.Combine(directory, "best-weights.json")), Json)
            ?? throw new InvalidDataException("Invalid checkpoint");
        if (file.FormatVersion != 1) throw new InvalidDataException("Unsupported checkpoint version");
        var c = file.Configuration;
        var model = new Network(c.Width, c.Blocks, c.Activation, c.Alpha, c.WeightSeed); model.Restore(file.Parameters);
        Console.WriteLine(JsonSerializer.Serialize(new
        {
            epoch = file.Epoch,
            validation = Evaluation.Measure(Motion.Generate(c.ValidationCount, c.ValidationSeed, c.Steps, c.H), model.Forward),
            test = Evaluation.Measure(Motion.Generate(c.TestCount, c.TestSeed, c.Steps, c.H), model.Forward)
        }, Json));
    }
    private static GradientView[] Diagnostics(Network model, Sample[] data)
    {
        int count = Math.Min(64, data.Length); var inputSquares = new double[model.Blocks.Length];
        model.ZeroGradients();
        // Accumulate mean parameter gradients, but measure per-sample hidden gradients
        // at scale one so their scale does not silently depend on diagnostics count.
        for (int i = 0; i < count; i++)
        {
            var s = data[i]; var y = model.Forward(s.Input);
            model.Backward(Loss.Gradient(y, s.Target, count));
            for (int b = 0; b < model.Blocks.Length; b++) inputSquares[b] += Math.Pow(model.Blocks[b].LastInputGradientRms * count, 2);
        }
        return model.Blocks.Select((block, b) =>
        {
            var p = new[] { block.First.Weights, block.First.Bias, block.Second.Weights, block.Second.Bias };
            return new GradientView(b, Math.Sqrt(p.Sum(x => x.Gradients.Sum(g => g * g)) / p.Sum(x => x.Gradients.Length)), Math.Sqrt(inputSquares[b] / count));
        }).ToArray();
    }
    private static void WriteRow(StreamWriter csv, EpochRecord r)
    {
        csv.WriteLine($"{r.Epoch},{r.Train.Mse:R},{r.Validation.Mse:R},{r.Train.MeanDistance:R},{r.Validation.MeanDistance:R},{r.TrainSeconds:R},{r.EvaluationSeconds:R}");
        csv.Flush();
    }
    private static object GitRevision()
    {
        try
        {
            static string Git(string arguments)
            {
                using var p = Process.Start(new ProcessStartInfo("git", arguments) { RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false });
                if (p is null) return "unavailable";
                string result = p.StandardOutput.ReadToEnd().Trim(); p.WaitForExit();
                return p.ExitCode == 0 ? result : "unavailable";
            }
            return new { revision = Git("rev-parse HEAD"), dirty = Git("status --porcelain").Length != 0 };
        }
        catch { return new { revision = "unavailable", dirty = true }; }
    }
}
