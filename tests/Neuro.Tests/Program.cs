using Neuro;

var tests = new (string Name, Action Run)[]
{
    ("Motion sequential step, zero horizon, normalization and RNG", TestData),
    ("Activation derivatives including extreme inputs and kinks", TestActivations),
    ("Dense forward and transpose gradients on hand example", TestDense),
    ("Residual alpha=0 identity and independent blocks", TestIdentity),
    ("Full-network parameter and input finite differences, all activations", TestNetworkGradients),
    ("Batch means, incomplete batches and finite differences", TestBatch),
    ("SGD and Adam reference steps", TestOptimizers),
    ("Affine fit on known mapping", TestAffine),
    ("Evaluation purity, snapshot restore and reproducible training", TestReproducibility),
    ("Learning a small affine control task", TestLearning)
};
int failed = 0;
foreach (var (name, run) in tests)
{
    try { run(); Console.WriteLine($"PASS {name}"); }
    catch (Exception e) { failed++; Console.Error.WriteLine($"FAIL {name}: {e.Message}"); }
}
Console.WriteLine($"{tests.Length - failed}/{tests.Length} groups passed");
return failed == 0 ? 0 : 1;

static void Near(double actual, double expected, double abs = 1e-7, double rel = 1e-4)
{
    if (!double.IsFinite(actual) || !double.IsFinite(expected) || Math.Abs(actual - expected) > abs + rel * Math.Max(Math.Abs(actual), Math.Abs(expected)))
        throw new Exception($"Expected {expected:G17}, got {actual:G17}");
}
static void Check(bool condition, string message) { if (!condition) throw new Exception(message); }
static void TestData()
{
    var step = Motion.Step(0.2, 0.7, 0.2);
    double x = 0.2 + 0.2 * Math.Sin(0.7);
    Near(step.X, x); Near(step.Y, 0.7 - 0.2 * Math.Sin(x));
    Near(Motion.End(2, -3, 0, 0.2)[0], 2); Near(Motion.End(2, -3, 20, 0)[1], -3);
    var s = Motion.Point(2, -3, 0, 0.2);
    Near(s.Input[0] * Math.PI, 2); Near(s.Target[1] * Math.PI, -3);
    var a = Motion.Generate(16, 11, 20, 0.2); var b = Motion.Generate(16, 11, 20, 0.2);
    Check(a.Zip(b).All(p => p.First.Input.SequenceEqual(p.Second.Input) && p.First.Target.SequenceEqual(p.Second.Target)), "Generator not repeatable");
    Check(!a[0].Input.SequenceEqual(Motion.Generate(16, 12, 20, 0.2)[0].Input), "Seeds not independent");
    var grid = Motion.Grid(3, 0, 0.2);
    Near(grid[0].Input[0], -1); Near(grid[^1].Input[1], 1); Near(grid[4].Input[0], 0);
}
static void TestActivations()
{
    foreach (var kind in Enum.GetValues<ActivationKind>())
        foreach (double x in new[] { -3.0, -0.2, 0.3, 2.0 })
        {
            const double e = 1e-6;
            Near(Activations.Derivative(kind, x), (Activations.Value(kind, x + e) - Activations.Value(kind, x - e)) / (2 * e));
        }
    Near(Activations.Derivative(ActivationKind.Relu, 0), 0);
    Near(Activations.Derivative(ActivationKind.LeakyRelu, 0), 0.01);
    foreach (double x in new[] { -1000.0, 1000.0 })
    {
        Check(double.IsFinite(Activations.Value(ActivationKind.Swish, x)), "Unstable Swish");
        Check(double.IsFinite(Activations.Derivative(ActivationKind.Swish, x)), "Unstable derivative");
    }
}
static void TestDense()
{
    var d = new Dense("d", 2, 2, new RandomStream(1));
    new double[] { 1, 2, 3, 4 }.CopyTo(d.Weights.Values, 0);
    new double[] { 0.5, -0.5 }.CopyTo(d.Bias.Values, 0);
    double[] x = [2, -1]; var y = d.Forward(x);
    Near(y[0], 0.5); Near(y[1], 1.5);
    var dx = d.Backward(x, [0.1, 0.2]);
    Near(dx[0], 0.7); Near(dx[1], 1);
    Near(d.Weights.Gradients[0], 0.2); Near(d.Weights.Gradients[3], -0.2); Near(d.Bias.Gradients[1], 0.2);
}
static void TestIdentity()
{
    var net = new Network(20, 10, ActivationKind.Swish, 0.1, 42);
    Check(net.ParameterCount == 8502, "Wrong parameter count");
    Check(!ReferenceEquals(net.Blocks[0].First.Weights.Values, net.Blocks[1].First.Weights.Values), "Shared weights");
    var block = new ResidualBlock(0, 3, ActivationKind.Swish, 0, new RandomStream(42));
    double[] input = [0.2, -0.5, 0.7], g = [0.1, -0.3, 0.4];
    Check(block.Forward(input).SequenceEqual(input), "Identity forward failed");
    Check(block.Backward(g).SequenceEqual(g), "Identity backward failed");
    foreach (var p in new[] { block.First.Weights, block.First.Bias, block.Second.Weights, block.Second.Bias })
        Check(p.Gradients.All(v => v == 0), "Branch gradients must be zero");
}
static void TestNetworkGradients()
{
    foreach (var kind in Enum.GetValues<ActivationKind>())
    {
        var net = new Network(3, 2, kind, 0.1, 123);
        double[] input = [0.3, -0.4], target = [-0.2, 0.6];
        net.ZeroGradients(); var y = net.Forward(input);
        var dx = (double[])net.Backward(Loss.Gradient(y, target, 1)).Clone();
        var gradients = net.Parameters.Select(p => (double[])p.Gradients.Clone()).ToArray();
        for (int p = 0; p < net.Parameters.Length; p++)
            for (int i = 0; i < net.Parameters[p].Values.Length; i++)
            {
                double original = net.Parameters[p].Values[i]; const double e = 1e-6;
                net.Parameters[p].Values[i] = original + e; double plus = Loss.Mse(net.Forward(input), target);
                net.Parameters[p].Values[i] = original - e; double minus = Loss.Mse(net.Forward(input), target);
                net.Parameters[p].Values[i] = original;
                Near(gradients[p][i], (plus - minus) / (2 * e));
            }
        for (int i = 0; i < 2; i++)
        {
            double original = input[i]; const double e = 1e-6;
            input[i] = original + e; double plus = Loss.Mse(net.Forward(input), target);
            input[i] = original - e; double minus = Loss.Mse(net.Forward(input), target);
            input[i] = original; Near(dx[i], (plus - minus) / (2 * e));
        }
    }
}
static void TestBatch()
{
    var net = new Network(3, 2, ActivationKind.Swish, 0.1, 12);
    var data = Motion.Generate(5, 33, 20, 0.2); int[] order = [4, 2, 0, 1, 3];
    var expected = net.Parameters.Select(p => new double[p.Values.Length]).ToArray();
    for (int j = 1; j < 4; j++)
    {
        var s = data[order[j]]; net.ZeroGradients(); var y = net.Forward(s.Input);
        net.Backward(Loss.Gradient(y, s.Target, 1));
        for (int p = 0; p < expected.Length; p++) for (int i = 0; i < expected[p].Length; i++) expected[p][i] += net.Parameters[p].Gradients[i] / 3;
    }
    double loss = Batches.Accumulate(net, data, order, 1, 3);
    for (int p = 0; p < expected.Length; p++) for (int i = 0; i < expected[p].Length; i++) Near(net.Parameters[p].Gradients[i], expected[p][i]);
    Near(loss, order.Skip(1).Take(3).Average(i => Loss.Mse(net.Forward(data[i].Input), data[i].Target)));
    var weight = net.Parameters[0]; double original = weight.Values[0], analytic = weight.Gradients[0];
    const double eps = 1e-6;
    weight.Values[0] = original + eps; double plus = order.Skip(1).Take(3).Average(i => Loss.Mse(net.Forward(data[i].Input), data[i].Target));
    weight.Values[0] = original - eps; double minus = order.Skip(1).Take(3).Average(i => Loss.Mse(net.Forward(data[i].Input), data[i].Target));
    weight.Values[0] = original; Near(analytic, (plus - minus) / (2 * eps));
    Batches.Accumulate(net, data, order, 4, 1);
    var tail = net.Parameters.Select(p => (double[])p.Gradients.Clone()).ToArray();
    net.ZeroGradients(); var sTail = data[order[4]]; var output = net.Forward(sTail.Input);
    net.Backward(Loss.Gradient(output, sTail.Target, 1));
    for (int p = 0; p < tail.Length; p++) for (int i = 0; i < tail[p].Length; i++) Near(tail[p][i], net.Parameters[p].Gradients[i]);
}
static void TestOptimizers()
{
    var p = new Parameter("p", 1); p.Values[0] = 1; p.Gradients[0] = 2;
    new Optimizer([p], OptimizerKind.Sgd, 0.1).Step(); Near(p.Values[0], 0.8, 1e-12, 0);
    p.Values[0] = 1; var adam = new Optimizer([p], OptimizerKind.Adam, 0.1);
    adam.Step(); Near(p.Values[0], 1 - 0.1 * 2 / (2 + 1e-8), 1e-12, 0);
    p.Gradients[0] = -1; adam.Step();
    double m = 0.9 * 0.2 - 0.1, v = 0.999 * 0.004 + 0.001;
    double expected = 1 - 0.1 * 2 / (2 + 1e-8) - 0.1 * (m / (1 - 0.81)) / (Math.Sqrt(v / (1 - 0.998001)) + 1e-8);
    Near(p.Values[0], expected, 1e-12, 0); Check(adam.StepCount == 2, "Step count");
}
static void TestAffine()
{
    var data = Motion.Generate(64, 2, 0, 0.2).Select(s => new Sample(s.Input,
        [2 * s.Input[0] - s.Input[1] + 0.3, s.Input[0] + 3 * s.Input[1] - 0.4])).ToArray();
    var affine = new AffineBaseline(data);
    Check(Evaluation.Measure(data, affine.Predict).Mse < 1e-25, "Affine fit incorrect");
}
static void TestReproducibility()
{
    var data = Motion.Generate(17, 2, 20, 0.2);
    var a = new Network(4, 2, ActivationKind.Swish, 0.1, 4);
    var b = new Network(4, 2, ActivationKind.Swish, 0.1, 4);
    var saved = a.Snapshot(); var gradients = a.Parameters.Select(p => (double[])p.Gradients.Clone()).ToArray();
    Evaluation.Measure(data, a.Forward);
    Check(saved.Zip(a.Snapshot()).All(pair => pair.First.SequenceEqual(pair.Second)), "Evaluation changes parameters");
    Check(gradients.Zip(a.Parameters).All(pair => pair.First.SequenceEqual(pair.Second.Gradients)), "Evaluation changes gradients");
    var oa = new Optimizer(a.Parameters, OptimizerKind.Adam, 0.001); var ob = new Optimizer(b.Parameters, OptimizerKind.Adam, 0.001);
    int[] order = Enumerable.Range(0, data.Length).ToArray();
    for (int i = 0; i < 3; i++)
    {
        Batches.Accumulate(a, data, order, 0, 17); oa.Step();
        Batches.Accumulate(b, data, order, 0, 17); ob.Step();
    }
    Check(a.Snapshot().Zip(b.Snapshot()).All(pair => pair.First.SequenceEqual(pair.Second)), "Training not repeatable");
    var newNet = new Network(4, 2, ActivationKind.Swish, 0.1, 5); newNet.Restore(a.Snapshot());
    Check(a.Forward(data[0].Input).SequenceEqual(newNet.Forward(data[0].Input)), "Restored predictions differ");
    a.Restore(saved); Check(a.Snapshot().Zip(saved).All(pair => pair.First.SequenceEqual(pair.Second)), "Restore failed");
}
static void TestLearning()
{
    var data = Motion.Generate(64, 123, 0, 0.2);
    var net = new Network(4, 2, ActivationKind.Swish, 0.1, 4);
    var optimizer = new Optimizer(net.Parameters, OptimizerKind.Adam, 0.01);
    int[] order = Enumerable.Range(0, data.Length).ToArray();
    double before = Evaluation.Measure(data, net.Forward).Mse;
    for (int i = 0; i < 300; i++) { Batches.Accumulate(net, data, order, 0, data.Length); optimizer.Step(); }
    double after = Evaluation.Measure(Motion.Generate(64, 321, 0, 0.2), net.Forward).Mse;
    Check(after < 0.001 && after < before * 0.01, $"Did not learn: {before} -> {after}");
}
