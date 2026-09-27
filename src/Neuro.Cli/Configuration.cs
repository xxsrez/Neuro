using System.Globalization;
using Neuro;

namespace Neuro.Cli;

public sealed record Configuration
{
    public int Width { get; init; } = 20;
    public int Blocks { get; init; } = 10;
    public ActivationKind Activation { get; init; } = ActivationKind.Swish;
    public OptimizerKind Optimizer { get; init; } = OptimizerKind.Adam;
    public double Alpha { get; init; } = 0.1;
    public double LearningRate { get; init; } = 0.001;
    public int Epochs { get; init; } = 200;
    public int BatchSize { get; init; } = 64;
    public int TrainCount { get; init; } = 8192;
    public int ValidationCount { get; init; } = 2048;
    public int TestCount { get; init; } = 2048;
    public int Steps { get; init; } = 20;
    public double H { get; init; } = 0.2;
    public int GridSide { get; init; } = 101;
    public ulong TrainSeed { get; init; } = 1001;
    public ulong ValidationSeed { get; init; } = 2002;
    public ulong TestSeed { get; init; } = 3003;
    public ulong WeightSeed { get; init; } = 42;
    public ulong ShuffleSeed { get; init; } = 4004;
    public string Output { get; init; } = "runs/prototype";
    public double AcceptanceMse { get; init; } = 0.01;

    public static Configuration Parse(string[] args)
    {
        var c = new Configuration();
        for (int i = 0; i < args.Length; i += 2)
        {
            if (i + 1 >= args.Length) throw new ArgumentException($"Missing value for {args[i]}");
            string v = args[i + 1];
            int Int() => int.Parse(v, CultureInfo.InvariantCulture);
            double Number() => double.Parse(v, CultureInfo.InvariantCulture);
            ulong Seed() => ulong.Parse(v, CultureInfo.InvariantCulture);
            c = args[i] switch
            {
                "--width" => c with { Width = Int() }, "--blocks" => c with { Blocks = Int() },
                "--activation" => c with { Activation = Enum.Parse<ActivationKind>(v, true) },
                "--optimizer" => c with { Optimizer = Enum.Parse<OptimizerKind>(v, true) },
                "--alpha" => c with { Alpha = Number() }, "--lr" => c with { LearningRate = Number() },
                "--epochs" => c with { Epochs = Int() }, "--batch" => c with { BatchSize = Int() },
                "--train" => c with { TrainCount = Int() }, "--validation" => c with { ValidationCount = Int() },
                "--test" => c with { TestCount = Int() }, "--steps" => c with { Steps = Int() },
                "--h" => c with { H = Number() }, "--grid" => c with { GridSide = Int() },
                "--weight-seed" => c with { WeightSeed = Seed() }, "--shuffle-seed" => c with { ShuffleSeed = Seed() },
                "--train-seed" => c with { TrainSeed = Seed() }, "--validation-seed" => c with { ValidationSeed = Seed() },
                "--test-seed" => c with { TestSeed = Seed() }, "--out" => c with { Output = v },
                _ => throw new ArgumentException($"Unknown option {args[i]}")
            };
        }
        c.Validate(); return c;
    }
    private void Validate()
    {
        if (Width is < 1 or > 256 || Blocks is < 0 or > 100 || Epochs is < 1 or > 10000 || BatchSize < 1 ||
            TrainCount is < 4 or > 1000000 || ValidationCount is < 1 or > 1000000 || TestCount is < 1 or > 1000000 ||
            Steps is < 0 or > 10000 || GridSide is < 2 or > 201)
            throw new ArgumentException("Invalid or excessive dimensions; see --help.");
        if (!Enum.IsDefined(Activation) || !Enum.IsDefined(Optimizer) || !double.IsFinite(Alpha) ||
            !double.IsFinite(H) || H < 0 || !double.IsFinite(LearningRate) || LearningRate <= 0)
            throw new ArgumentException("Invalid activation, optimizer or numeric setting.");
        if (new[] { TrainSeed, ValidationSeed, TestSeed }.Distinct().Count() != 3)
            throw new ArgumentException("Use distinct seeds for train, validation and test.");
    }
}
