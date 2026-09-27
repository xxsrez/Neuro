namespace Neuro;

public enum ActivationKind { Swish, Relu, Tanh, LeakyRelu }

public static class Activations
{
    public static double Sigmoid(double x)
    {
        if (x >= 0) return 1 / (1 + Math.Exp(-x));
        double t = Math.Exp(x);
        return t / (1 + t);
    }
    public static double Value(ActivationKind kind, double x) => kind switch
    {
        ActivationKind.Swish => x * Sigmoid(x),
        ActivationKind.Relu => Math.Max(0, x),
        ActivationKind.Tanh => Math.Tanh(x),
        ActivationKind.LeakyRelu => x >= 0 ? x : 0.01 * x,
        _ => throw new ArgumentOutOfRangeException(nameof(kind))
    };
    public static double Derivative(ActivationKind kind, double x)
    {
        switch (kind)
        {
            case ActivationKind.Swish:
                double s = Sigmoid(x);
                return s + x * s * (1 - s);
            case ActivationKind.Relu: return x > 0 ? 1 : 0;
            case ActivationKind.Tanh:
                double t = Math.Tanh(x);
                return 1 - t * t;
            case ActivationKind.LeakyRelu: return x > 0 ? 1 : 0.01;
            default: throw new ArgumentOutOfRangeException(nameof(kind));
        }
    }
}
