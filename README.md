🧠 SimpleAI: Lightweight Neural Network Library in C#

This is a minimal educational project implementing a basic multi-layer neural network (MLP) in C#. It supports:

    fully connected architecture;

    customizable activation functions;

    backpropagation-based training;

    training statistics and weight inspection.

📁 Project Structure

    📦 NetworkLibrary — core neural network logic:

        NeuralNetwork.cs — main network implementation

        Neuron.cs — individual neuron logic

        ActivationType.cs — supported activation functions

        TrainingStats.cs — tracks training progress

        RandomExtensions.cs — helper methods

        DataGeneration.cs — utility to generate input data

    📦 MainApp — console demo project:

        Program.cs — entry point for training and testing

🔧 Example Usage

using NetworkLibrary;

int[] layers = { 2, 3, 1 };
var network = new NeuralNetwork(
    layerSizes: layers,
    learningRate: 0.1,
    seed: 42,
    hidden: ActivationType.Sigmoid,
    output: ActivationType.Sigmoid
);

// XOR training set
var data = new List<(double[], double[])>
{
    (new double[] { 0, 0 }, new double[] { 0 }),
    (new double[] { 0, 1 }, new double[] { 1 }),
    (new double[] { 1, 0 }, new double[] { 1 }),
    (new double[] { 1, 1 }, new double[] { 0 })
};

var stats = network.TrainBatch(data, epochs: 5000, targetError: 0.01, printInterval: 500);

📈 Training Output

The returned TrainingStats object includes:

    FinalError — the final average error after training

    EpochsToConverge — number of epochs until target error was reached

    ErrorHistory — error for each epoch (suitable for plotting)

⚙️ Build & Run

    Requires .NET 6 or newer

    Open the solution in JetBrains Rider or Visual Studio

    Or use the command line:

dotnet build
dotnet run --project MainApp

🪪 License

MIT License — free for personal and commercial use. Attribution is appreciated but not required.
👨‍💻 Author

Victor Herasymenko
https://github.com/hslt0
