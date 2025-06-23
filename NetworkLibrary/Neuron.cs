namespace NetworkLibrary;

public class Neuron
{
    public readonly double[] Weights;
    
    private double bias;
    private ActivationType ActivationType { get; }

    private double LastOutput { get; set; }
    public double LastWeightedInput { get; private set; }

    public Neuron(int inputCount, Random random, ActivationType activationType = ActivationType.Sigmoid)
    {
        Weights = new double[inputCount];
        ActivationType = activationType;
            
        double scale = ActivationType == ActivationType.ReLu 
            ? Math.Sqrt(2.0 / inputCount)  
            : Math.Sqrt(2.0 / (inputCount + 1));
                
        for (int i = 0; i < inputCount; i++)
        {
            Weights[i] = random.NextGaussian() * scale;
        }
        bias = 0;
    }

    public double ProcessInput(double[] inputs)
    {
        if (inputs.Length != Weights.Length)
            throw new ArgumentException("Input count doesn't match weights count.");

        double sum = inputs.Select((t, i) => t * Weights[i]).Sum();
        sum += bias;
        LastWeightedInput = sum;
        LastOutput = ActivationFunction(sum);
        return LastOutput;
    }

    public void UpdateWeights(double[] weightDeltas, double biasDelta)
    {
        for (int i = 0; i < Weights.Length; i++)
        {
            Weights[i] += weightDeltas[i];
        }
        bias += biasDelta;
    }

    private double ActivationFunction(double x)
    {
        return ActivationType switch
        {
            ActivationType.Linear => x,
            ActivationType.ReLu => Math.Max(0, x),
            ActivationType.Sigmoid => SigmoidFunction(x),
            ActivationType.Tanh => TanhFunction(x),
            _ => SigmoidFunction(x)
        };
    }

    private double SigmoidFunction(double x)
    {
        try
        {
            if (x > 500) return 1.0;
            if (x < -500) return 0.0;
            double exp = Math.Exp(-x);
            return 1.0 / (1.0 + exp);
        }
        catch (OverflowException)
        {
            return x > 0 ? 1.0 : 0.0;
        }
    }

    private static double TanhFunction(double x)
    {
        try
        {
            if (x > 500) return 1.0;
            if (x < -500) return -1.0;
            return Math.Tanh(x);
        }
        catch (OverflowException)
        {
            return x > 0 ? 1.0 : -1.0;
        }
    }

    public double ActivationDerivative(double x)
    {
        return ActivationType switch
        {
            ActivationType.Linear => 1.0,
            ActivationType.ReLu => x > 0 ? 1.0 : 0.0,
            ActivationType.Sigmoid => SigmoidDerivative(x),
            ActivationType.Tanh => TanhDerivative(x),
            _ => SigmoidDerivative(x)
        };
    }

    private static double SigmoidDerivative(double x)
    {
        double sigmoid = new Neuron(1, new Random()).SigmoidFunction(x);
        return sigmoid * (1 - sigmoid);
    }

    private static double TanhDerivative(double x)
    {
        double tanh = TanhFunction(x);
        return 1 - tanh * tanh;
    }

    public override string ToString()
    {
        return $"Weights: [{string.Join(", ", Weights.Select(w => w.ToString("F4")))}], Bias: {bias:F6}, Activation: {ActivationType}";
    }
}