namespace Neuro;

public sealed class Parameter(string name, int count)
{
    public string Name { get; } = name;
    public double[] Values { get; } = new double[count];
    public double[] Gradients { get; } = new double[count];
}

// Weight storage: W[output * inputSize + input]. All outputs and backward
// buffers are reused. A model instance is single-threaded and caches one forward.
public sealed class Dense
{
    public int InputSize { get; }
    public int OutputSize { get; }
    public Parameter Weights { get; }
    public Parameter Bias { get; }
    public double[] Output { get; }
    private readonly double[] inputGradient;
    public Dense(string name, int inputSize, int outputSize, RandomStream rng)
    {
        if (inputSize < 1 || outputSize < 1) throw new ArgumentOutOfRangeException(nameof(inputSize));
        InputSize = inputSize; OutputSize = outputSize;
        Weights = new(name + ".weight", inputSize * outputSize);
        Bias = new(name + ".bias", outputSize);
        Output = new double[outputSize]; inputGradient = new double[inputSize];
        double limit = Math.Sqrt(6.0 / (inputSize + outputSize));
        for (int i = 0; i < Weights.Values.Length; i++) Weights.Values[i] = (2 * rng.Uniform() - 1) * limit;
    }
    public double[] Forward(double[] input)
    {
        if (input.Length != InputSize) throw new ArgumentException("Wrong input dimension.");
        for (int o = 0; o < OutputSize; o++)
        {
            double sum = Bias.Values[o];
            int offset = o * InputSize;
            for (int i = 0; i < InputSize; i++) sum += Weights.Values[offset + i] * input[i];
            Output[o] = sum;
        }
        return Output;
    }
    public double[] Backward(double[] input, double[] gradient)
    {
        if (input.Length != InputSize || gradient.Length != OutputSize) throw new ArgumentException("Wrong gradient dimension.");
        Array.Clear(inputGradient);
        for (int o = 0; o < OutputSize; o++)
        {
            double g = gradient[o]; int offset = o * InputSize;
            Bias.Gradients[o] += g;
            for (int i = 0; i < InputSize; i++)
            {
                Weights.Gradients[offset + i] += g * input[i];
                inputGradient[i] += g * Weights.Values[offset + i];
            }
        }
        return inputGradient;
    }
}

public sealed class ResidualBlock
{
    public Dense First { get; }
    public Dense Second { get; }
    public double[] Activated { get; }
    public double[] Output { get; }
    public double LastInputGradientRms { get; private set; }
    private readonly double[] branchGradient;
    private readonly double[] preGradient;
    private readonly double[] inputGradient;
    private double[]? cachedInput;
    private readonly ActivationKind activation;
    private readonly double alpha;
    public ResidualBlock(int index, int width, ActivationKind activation, double alpha, RandomStream rng)
    {
        this.activation = activation; this.alpha = alpha;
        First = new($"block{index}.first", width, width, rng);
        Second = new($"block{index}.second", width, width, rng);
        Activated = new double[width]; Output = new double[width];
        branchGradient = new double[width]; preGradient = new double[width]; inputGradient = new double[width];
    }
    public double[] Forward(double[] input)
    {
        cachedInput = input;
        var a = First.Forward(input);
        for (int i = 0; i < input.Length; i++) Activated[i] = Activations.Value(activation, a[i]);
        var r = Second.Forward(Activated);
        for (int i = 0; i < input.Length; i++) Output[i] = input[i] + alpha * r[i];
        return Output;
    }
    public double[] Backward(double[] gradient)
    {
        if (cachedInput is null) throw new InvalidOperationException("Forward required before backward.");
        for (int i = 0; i < gradient.Length; i++) branchGradient[i] = alpha * gradient[i];
        var qGradient = Second.Backward(Activated, branchGradient);
        for (int i = 0; i < gradient.Length; i++)
            preGradient[i] = qGradient[i] * Activations.Derivative(activation, First.Output[i]);
        var branchInput = First.Backward(cachedInput, preGradient);
        double squares = 0;
        for (int i = 0; i < gradient.Length; i++)
        {
            inputGradient[i] = gradient[i] + branchInput[i]; // Identity route.
            squares += inputGradient[i] * inputGradient[i];
        }
        LastInputGradientRms = Math.Sqrt(squares / gradient.Length);
        return inputGradient;
    }
}

public sealed class Network
{
    public Dense Input { get; }
    public Dense Output { get; }
    public ResidualBlock[] Blocks { get; }
    public Parameter[] Parameters { get; }
    public int ParameterCount => Parameters.Sum(p => p.Values.Length);
    private double[]? cachedInput;
    public Network(int width, int blocks, ActivationKind activation, double alpha, ulong seed)
    {
        if (blocks < 0 || !double.IsFinite(alpha)) throw new ArgumentOutOfRangeException(nameof(blocks));
        var rng = new RandomStream(seed);
        Input = new("input", 2, width, rng);
        Blocks = Enumerable.Range(0, blocks).Select(i => new ResidualBlock(i, width, activation, alpha, rng)).ToArray();
        Output = new("output", width, 2, rng);
        var parameters = new List<Parameter> { Input.Weights, Input.Bias };
        foreach (var block in Blocks) parameters.AddRange([block.First.Weights, block.First.Bias, block.Second.Weights, block.Second.Bias]);
        parameters.AddRange([Output.Weights, Output.Bias]); Parameters = parameters.ToArray();
    }
    public double[] Forward(double[] input)
    {
        cachedInput = input;
        var z = Input.Forward(input);
        foreach (var block in Blocks) z = block.Forward(z);
        return Output.Forward(z);
    }
    public double[] Backward(double[] gradient)
    {
        if (cachedInput is null) throw new InvalidOperationException("Forward required before backward.");
        var z = Blocks.Length == 0 ? Input.Output : Blocks[^1].Output;
        var g = Output.Backward(z, gradient);
        for (int i = Blocks.Length - 1; i >= 0; i--) g = Blocks[i].Backward(g);
        return Input.Backward(cachedInput, g);
    }
    public void ZeroGradients() { foreach (var p in Parameters) Array.Clear(p.Gradients); }
    public double[][] Snapshot() => Parameters.Select(p => (double[])p.Values.Clone()).ToArray();
    public void Restore(double[][] values)
    {
        if (values.Length != Parameters.Length || values.Where((v, i) => v.Length != Parameters[i].Values.Length).Any())
            throw new ArgumentException("Wrong parameter shape.");
        foreach (var array in values) if (array.Any(v => !double.IsFinite(v))) throw new ArgumentException("Non-finite weights.");
        for (int i = 0; i < values.Length; i++) values[i].CopyTo(Parameters[i].Values, 0);
    }
}

public static class Loss
{
    public static double Mse(double[] prediction, double[] target)
    {
        if (prediction.Length != 2 || target.Length != 2) throw new ArgumentException("Expected two coordinates.");
        double x = prediction[0] - target[0], y = prediction[1] - target[1];
        return (x * x + y * y) / 2;
    }
    public static double[] Gradient(double[] prediction, double[] target, int batchSize)
    {
        if (batchSize < 1) throw new ArgumentOutOfRangeException(nameof(batchSize));
        return [(prediction[0] - target[0]) / batchSize, (prediction[1] - target[1]) / batchSize];
    }
}
