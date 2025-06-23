namespace NetworkLibrary;

public class NeuralNetwork
    {
        private readonly List<List<Neuron>> layers;
        
        private readonly Random random;
        private double LearningRate { get; }
        private double LastError { get; set; }
        private int[] LayerSizes { get; }

        public NeuralNetwork(int[] layerSizes, double learningRate, int? seed, ActivationType hidden, ActivationType output)
        {
            if (layerSizes.Length < 2)
                throw new ArgumentException("Network must have at least 2 layers");

            LayerSizes = layerSizes;
            LearningRate = learningRate;
            random = seed.HasValue ? new Random(seed.Value) : new Random();
            layers = [];

            InitializeLayers(hidden, output);
        }

        private void InitializeLayers(ActivationType hiddenActivation, ActivationType outputActivation)
        {
            layers.Clear();
            
            for (int i = 0; i < LayerSizes.Length; i++)
            {
                var layer = new List<Neuron>();
                
                if (i == 0)
                {
                    layers.Add(layer);
                    continue;
                }
                
                var activationType = i == LayerSizes.Length - 1 ? outputActivation : hiddenActivation;
                
                for (int j = 0; j < LayerSizes[i]; j++)
                {
                    layer.Add(new Neuron(LayerSizes[i - 1], random, activationType));
                }
                layers.Add(layer);
            }
        }

        public double[] ProcessInput(double[] inputs)
        {
            double[] currentInputs = inputs;

            for (int layerIndex = 1; layerIndex < layers.Count; layerIndex++)
            {
                var layer = layers[layerIndex];
                double[] outputs = new double[layer.Count];

                for (int i = 0; i < layer.Count; i++)
                {
                    outputs[i] = layer[i].ProcessInput(currentInputs);
                }

                currentInputs = outputs;
            }

            return currentInputs;
        }

        private void Train(double[] inputs, double[] expectedOutputs)
        {
            if (expectedOutputs.Length != layers.Last().Count)
                throw new ArgumentException("Expected output size doesn't match output layer size.");

            var outputs = new List<double[]>();
            double[] currentInputs = inputs;

            outputs.Add(currentInputs);

            for (int layerIndex = 1; layerIndex < layers.Count; layerIndex++)
            {
                var layer = layers[layerIndex];
                double[] layerOutputs = new double[layer.Count];
                for (int i = 0; i < layer.Count; i++)
                {
                    layerOutputs[i] = layer[i].ProcessInput(currentInputs);
                }
                outputs.Add(layerOutputs);
                currentInputs = layerOutputs;
            }

            double[] errors = new double[expectedOutputs.Length];
            double totalError = 0;
            for (int i = 0; i < expectedOutputs.Length; i++)
            {
                errors[i] = expectedOutputs[i] - outputs.Last()[i];
                totalError += errors[i] * errors[i];
            }

            LastError = Math.Sqrt((totalError / errors.Length));

            double[][] layerErrors = new double[layers.Count][];
            
            for (int i = 0; i < layers.Count; i++)
            {
                if (i == 0)
                    layerErrors[i] = new double[inputs.Length];
                else
                    layerErrors[i] = new double[layers[i].Count];
            }
            
            layerErrors[^1] = errors;

            for (int layerIndex = layers.Count - 2; layerIndex >= 1; layerIndex--)
            {
                for (int i = 0; i < layers[layerIndex].Count; i++)
                {
                    double errorSum = 0;
                    for (int j = 0; j < layers[layerIndex + 1].Count; j++)
                    {
                        var nextNeuron = layers[layerIndex + 1][j];
                        errorSum += nextNeuron.Weights[i] * layerErrors[layerIndex + 1][j];
                    }
                    layerErrors[layerIndex][i] = errorSum;
                }
            }

            for (int layerIndex = 1; layerIndex < layers.Count; layerIndex++)
            {
                var layer = layers[layerIndex];
                var inputsToUse = outputs[layerIndex - 1];

                for (int neuronIndex = 0; neuronIndex < layer.Count; neuronIndex++)
                {
                    var neuron = layer[neuronIndex];
                    double derivative = neuron.ActivationDerivative(neuron.LastWeightedInput);
                    double errorSignal = layerErrors[layerIndex][neuronIndex] * derivative;

                    double[] weightDeltas = new double[neuron.Weights.Length];
                    for (int w = 0; w < neuron.Weights.Length; w++)
                    {
                        weightDeltas[w] = LearningRate * errorSignal * inputsToUse[w];
                    }

                    double biasDelta = LearningRate * errorSignal;
                    neuron.UpdateWeights(weightDeltas, biasDelta);
                }
            }
        }

        public void PrintWeights()
        {
            for (int i = 1; i < layers.Count; i++)
            {
                Console.WriteLine($"Layer {i}:");
                for (int j = 0; j < layers[i].Count; j++)
                {
                    Console.WriteLine($"  Neuron {j + 1}: {layers[i][j]}");
                }
            }
        }

        public TrainingStats TrainBatch(List<(double[] inputs, double[] outputs)> trainingData,
                                        int epochs, double targetError, int printInterval)
        {
            var stats = new TrainingStats();

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double epochError = 0;
                var shuffledData = trainingData.OrderBy(_ => random.Next()).ToList();

                foreach (var (inputs, outputs) in shuffledData)
                {
                    Train(inputs, outputs);
                    epochError += LastError * LastError;
                }

                epochError = Math.Sqrt((epochError / trainingData.Count));
                stats.ErrorHistory.Add(epochError);

                if (epoch % printInterval == 0 || epoch == epochs - 1)
                    Console.WriteLine($"Epoch: {epoch + 1}, Average Error: {epochError:F6}");

                if (!(epochError <= targetError)) continue;
                
                Console.WriteLine($"Target error reached at epoch {epoch + 1}");
                stats.EpochsToConverge = epoch + 1;
                break;
            }

            stats.FinalError = stats.ErrorHistory.Last();
            return stats;
        }
    }