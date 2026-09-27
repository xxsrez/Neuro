namespace Neuro;

public enum OptimizerKind { Adam, Sgd }

public sealed class Optimizer
{
    private readonly Parameter[] parameters;
    private readonly double[][] first, second;
    private readonly OptimizerKind kind;
    private readonly double learningRate;
    public long StepCount { get; private set; }
    public Optimizer(Parameter[] parameters, OptimizerKind kind, double learningRate)
    {
        if (!double.IsFinite(learningRate) || learningRate <= 0) throw new ArgumentOutOfRangeException(nameof(learningRate));
        if (!Enum.IsDefined(kind)) throw new ArgumentOutOfRangeException(nameof(kind));
        this.parameters = parameters; this.kind = kind; this.learningRate = learningRate;
        first = parameters.Select(p => new double[p.Values.Length]).ToArray();
        second = parameters.Select(p => new double[p.Values.Length]).ToArray();
    }
    public void Step()
    {
        StepCount++;
        double correction1 = 1 - Math.Pow(0.9, StepCount), correction2 = 1 - Math.Pow(0.999, StepCount);
        for (int p = 0; p < parameters.Length; p++)
            for (int i = 0; i < parameters[p].Values.Length; i++)
            {
                double g = parameters[p].Gradients[i];
                if (!double.IsFinite(g)) throw new ArithmeticException($"Non-finite gradient: {parameters[p].Name}[{i}]");
                double update;
                if (kind == OptimizerKind.Sgd) update = learningRate * g;
                else
                {
                    first[p][i] = 0.9 * first[p][i] + 0.1 * g;
                    second[p][i] = 0.999 * second[p][i] + 0.001 * g * g;
                    if (!double.IsFinite(first[p][i]) || !double.IsFinite(second[p][i])) throw new ArithmeticException("Non-finite Adam state.");
                    update = learningRate * (first[p][i] / correction1) / (Math.Sqrt(second[p][i] / correction2) + 1e-8);
                }
                parameters[p].Values[i] -= update;
                if (!double.IsFinite(parameters[p].Values[i])) throw new ArithmeticException("Non-finite parameter after update.");
            }
    }
}

public sealed record Metrics(double Mse, double MeanDistance);

public static class Evaluation
{
    public static Metrics Measure(Sample[] data, Func<double[], double[]> predict)
    {
        double loss = 0, distance = 0;
        foreach (var sample in data)
        {
            var output = predict(sample.Input);
            double mse = Loss.Mse(output, sample.Target);
            if (!double.IsFinite(mse)) throw new ArithmeticException("Non-finite evaluation.");
            loss += mse; distance += Math.Sqrt(2 * mse) * Math.PI;
        }
        return new(loss / data.Length, distance / data.Length);
    }
}

public static class Batches
{
    public static double Accumulate(Network model, Sample[] data, int[] order, int start, int count)
    {
        if (count < 1 || start < 0 || start + count > order.Length) throw new ArgumentOutOfRangeException(nameof(count));
        model.ZeroGradients(); double loss = 0;
        for (int j = start; j < start + count; j++)
        {
            var s = data[order[j]]; var output = model.Forward(s.Input);
            loss += Loss.Mse(output, s.Target);
            model.Backward(Loss.Gradient(output, s.Target, count));
        }
        return loss / count;
    }
}

// Fit an affine control y = Ax + b to training data only using a tiny
// normal equation (3 features: x, y, 1), solved here with pivoted elimination.
public sealed class AffineBaseline
{
    public double[][] Coefficients { get; }
    public AffineBaseline(Sample[] data)
    {
        var gram = new double[3, 3]; var rhs = new double[3, 2];
        foreach (var s in data)
        {
            double[] f = [s.Input[0], s.Input[1], 1];
            for (int r = 0; r < 3; r++)
            {
                for (int c = 0; c < 3; c++) gram[r, c] += f[r] * f[c];
                for (int c = 0; c < 2; c++) rhs[r, c] += f[r] * s.Target[c];
            }
        }
        var a = new double[3, 5];
        for (int r = 0; r < 3; r++)
        {
            for (int c = 0; c < 3; c++) a[r, c] = gram[r, c];
            for (int c = 0; c < 2; c++) a[r, 3 + c] = rhs[r, c];
        }
        for (int k = 0; k < 3; k++)
        {
            int pivot = k;
            for (int r = k + 1; r < 3; r++) if (Math.Abs(a[r, k]) > Math.Abs(a[pivot, k])) pivot = r;
            if (Math.Abs(a[pivot, k]) < 1e-12) throw new InvalidOperationException("Singular affine training data.");
            for (int c = 0; c < 5; c++) (a[k, c], a[pivot, c]) = (a[pivot, c], a[k, c]);
            double divisor = a[k, k];
            for (int c = k; c < 5; c++) a[k, c] /= divisor;
            for (int r = 0; r < 3; r++) if (r != k)
            {
                double factor = a[r, k];
                for (int c = k; c < 5; c++) a[r, c] -= factor * a[k, c];
            }
        }
        Coefficients = [Enumerable.Range(0, 3).Select(i => a[i, 3]).ToArray(), Enumerable.Range(0, 3).Select(i => a[i, 4]).ToArray()];
    }
    public double[] Predict(double[] input) => Coefficients.Select(c => c[0] * input[0] + c[1] * input[1] + c[2]).ToArray();
}
