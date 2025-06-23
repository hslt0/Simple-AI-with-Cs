namespace NetworkLibrary;

public static class DataGenerator
{
    public static List<(double[] inputs, double[] outputs)> GenerateImprovedConversionData(int count, double conversionFactor)
    {
        var random = new Random(42);
        var data = new List<(double[] inputs, double[] outputs)>();
            
        for (int i = 0; i < count; i++)
        {
            double randVal = random.NextDouble();
            double input = randVal switch
            {
                < 0.3 => 0.1 + random.NextDouble() * 9.9,
                < 0.6 => 10 + random.NextDouble() * 90,
                < 0.9 => 100 + random.NextDouble() * 400,
                _ => 500 + random.NextDouble() * 500
            };

            double output = input * conversionFactor;
                
            double normalizedInput = input / 1000;
            double normalizedOutput = output / (1000 * conversionFactor);
                
            data.Add(([normalizedInput], [normalizedOutput]));
        }
            
        return data;
    }
    
    public static List<(double[] inputs, double[] outputs)> GenerateSinExpData(int sampleCount, bool addNoise = false)
    {
        var data = new List<(double[] inputs, double[] outputs)>();
        var random = new Random(42);

        for (int i = 0; i < sampleCount; i++)
        {
            double x;
            if (i < sampleCount * 0.6) 
            {
                x = -1.5 + 3.0 * i / (sampleCount * 0.6 - 1);
            }
            else 
            {
                double t = (i - sampleCount * 0.6) / (sampleCount * 0.4 - 1);
                x = t < 0.5 ? -2.0 + 0.5 * t * 2 : 1.5 + 0.5 * (t - 0.5) * 2;
            }

            double y = Math.Sin(5 * x) * Math.Exp(-x * x);
        
            if (addNoise)
            {
                y += (random.NextDouble() - 0.5) * 0.01;
            }

            data.Add(([x], [y]));
        }

        for (int i = data.Count - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (data[i], data[j]) = (data[j], data[i]);
        }

        return data;
    }
}