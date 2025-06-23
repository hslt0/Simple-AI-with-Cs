using NetworkLibrary;

namespace MainApp;

public static class Program
{
    private const double KmToMilesConversion = 0.621371;
    private const int TrainingDataSize = 10000;
    
    private static void Main()
    {
        TrainModel(DataGenerator.GenerateImprovedConversionData(TrainingDataSize, KmToMilesConversion), "km_to_miles");
        TrainModel(DataGenerator.GenerateSinExpData(TrainingDataSize), "sin exp");
    }
    
    private static void TrainModel(List<(double[] input, double[] output)> data, string taskType)
{
    const int maxEpochs = 2000;
    const double targetError = 0.0003;

    Console.WriteLine("learning rate");
    if (!double.TryParse(Console.ReadLine(), out double learningRate) || learningRate <= 0)
    {
        learningRate = 0.15;
        Console.WriteLine($"Error, learning rate: {learningRate}");
    }
    
    (ActivationType hiden, ActivationType output) combination = taskType switch
    {
        "km_to_miles" => (ActivationType.Linear, ActivationType.Linear),
        "sin exp" => (ActivationType.Tanh, ActivationType.Linear),
        _ => throw new ArgumentException("Unknown task type")
    };

    var network = new NeuralNetwork([1, 15, 10, 1], learningRate, seed: 42, combination.hiden, combination.output);

    int splitIndex = (int)(data.Count * 0.8);
    var trainingData = data.Take(splitIndex).ToList();
    var validationData = data.Skip(splitIndex).ToList();

    Console.WriteLine($"training data: {trainingData.Count}, valid data: {validationData.Count}");

    Console.WriteLine($"Hidden layers: {combination.hiden}");
    Console.WriteLine($"Output layers: {combination.output}");

    Console.WriteLine("Start weights:");
    network.PrintWeights();
    Console.WriteLine();

    Console.WriteLine($"Learning on {trainingData.Count} examples...");
    var stats = network.TrainBatch(trainingData, maxEpochs, targetError, 100);

    Console.WriteLine($"Finale error: {stats.FinalError:F6}");
    if (stats.EpochsToConverge > 0)
        Console.WriteLine($"Wasted epoches: {stats.EpochsToConverge}");

    Console.WriteLine("Final weights: ");
    network.PrintWeights();

    Console.WriteLine("=== Testing ===");

    double[] testCases = taskType switch
    {
        "km_to_miles" => [0.1, 0.5, 1, 5, 10, 50, 100, 200, 500, 750, 999],
        "sin exp" => [0.1, 0.5, 1, 1.5, 2, 2.5, 3],
        _ => throw new ArgumentException("Unknown task type")
    };

    foreach (double input in testCases)
    {
        double normalizedInput = taskType switch
        {
            "km_to_miles" => input / 1000,
            "sin exp" => input / 2,
            _ => throw new ArgumentException("Unknown task type")
        };

        double normalizedPrediction = network.ProcessInput([normalizedInput])[0];

        double denormalizedPrediction = taskType switch
        {
            "km_to_miles" => normalizedPrediction * (1000 * KmToMilesConversion),
            "sin exp" => normalizedPrediction * 0.9,
            _ => throw new ArgumentException("Unknown task type")
        };

        double actual = taskType switch
        {
            "km_to_miles" => input * KmToMilesConversion,
            "sin exp" => Math.Sin(5 * input) * Math.Exp(-input * input),
            _ => throw new ArgumentException("Unknown task type")
        };

        double error = Math.Abs(denormalizedPrediction - actual);
        double errorPercent = actual != 0 ? (error / Math.Abs(actual)) * 100 : 0;

        string status = errorPercent < 1 ? "Excellent" :
                        errorPercent < 5 ? "Good" :
                        errorPercent < 10 ? "50/50" : "Can be better";

        Console.WriteLine($"{input:F2} -> {denormalizedPrediction:F6} (wanted: {actual:F6}, error: {errorPercent:F2}%) {status}");
    }

    Console.WriteLine("Learning statistics:");
    Console.WriteLine($"Epoches: {stats.ErrorHistory.Count}");
    Console.WriteLine($"Start error: {stats.ErrorHistory[0]:F6}");
    Console.WriteLine($"Final error: {stats.FinalError:F6}");
    Console.WriteLine($"Improvement: {(stats.ErrorHistory[0] - stats.FinalError):F6}");
}

}