namespace Neuro;

// Separate seeded instances isolate data, initialization and shuffle streams.
// SplitMix64 makes the stream independent of framework Random implementation.
public sealed class RandomStream(ulong seed)
{
    public ulong State { get; private set; } = seed;
    public ulong Next()
    {
        ulong z = State += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }
    public double Uniform() => (Next() >> 11) * (1.0 / (1UL << 53));
    public void Shuffle(int[] indices)
    {
        for (int i = indices.Length - 1; i > 0; i--)
        {
            int j = (int)(Uniform() * (i + 1));
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }
    }
}

public sealed record Sample(double[] Input, double[] Target);

public static class Motion
{
    public const string Version = "sequential-sine-v1";
    public static (double X, double Y) Step(double x, double y, double h)
    {
        double nextX = x + h * Math.Sin(y);
        return (nextX, y - h * Math.Sin(nextX));
    }
    public static double[] End(double x, double y, int steps, double h)
    {
        if (steps < 0 || !double.IsFinite(h)) throw new ArgumentOutOfRangeException(nameof(steps));
        for (int i = 0; i < steps; i++) (x, y) = Step(x, y, h);
        return [x, y];
    }
    public static Sample Point(double x, double y, int steps, double h)
    {
        var end = End(x, y, steps, h);
        return new([x / Math.PI, y / Math.PI], [end[0] / Math.PI, end[1] / Math.PI]);
    }
    public static Sample[] Generate(int count, ulong seed, int steps, double h)
    {
        if (count < 1) throw new ArgumentOutOfRangeException(nameof(count));
        var rng = new RandomStream(seed);
        return Enumerable.Range(0, count).Select(_ => Point(
            (2 * rng.Uniform() - 1) * Math.PI, (2 * rng.Uniform() - 1) * Math.PI, steps, h)).ToArray();
    }
    // Rows go from y=-pi to y=pi; map renderers explicitly invert the screen y axis.
    public static Sample[] Grid(int side, int steps, double h)
    {
        if (side < 2) throw new ArgumentOutOfRangeException(nameof(side));
        return Enumerable.Range(0, side * side).Select(i => Point(
            Math.PI * (2.0 * (i % side) / (side - 1) - 1),
            Math.PI * (2.0 * (i / side) / (side - 1) - 1), steps, h)).ToArray();
    }
}
