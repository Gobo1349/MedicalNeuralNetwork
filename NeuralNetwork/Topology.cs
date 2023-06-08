namespace NeuralNetwork;

public class Topology // описание сети 
{
    public int InputCount { get; } // количество входов в сеть 
    public int OutPutCount { get; } // количество выходов 
    public double LearningRate { get; } // характеристика скорости и качества обучения
    public List<int> HiddenLayers { get; } // количество нейронов в промежуточных слоях 

    public Topology(int inputCount, int outPutCount, double learningRate, params int[] layers) // params - ?, 3- количество нейронов в промежуточных слоях 
    {
        InputCount = inputCount;
        OutPutCount = outPutCount;
        LearningRate = learningRate;
        HiddenLayers = new List<int>();
        HiddenLayers.AddRange(layers);
    }
}