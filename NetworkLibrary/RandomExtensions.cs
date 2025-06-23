namespace NetworkLibrary;

public static class RandomExtensions
{
    private static bool hasSpare;
    private static double spare;

    public static double NextGaussian(this Random random, double mean = 0.0, double stdDev = 1.0)
    {
        if (hasSpare)
        {
            hasSpare = false;
            return spare * stdDev + mean;
        }

        hasSpare = true;
        double u = random.NextDouble();
        double v = random.NextDouble();
        double mag = stdDev * Math.Sqrt(-2.0 * Math.Log(u));
        spare = mag * Math.Cos(2.0 * Math.PI * v);
        return mag * Math.Sin(2.0 * Math.PI * v) + mean;
    }
}