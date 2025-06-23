namespace NetworkLibrary;

public class TrainingStats
{
    public List<double> ErrorHistory { get; } = [];
    public double FinalError { get; set; }
    public int EpochsToConverge { get; set; }
}