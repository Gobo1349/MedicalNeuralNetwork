namespace NeuralNetwork;

public class Layer
{
    private readonly NeuronType _type;

    // слой - набор нейронов 
    public List<Neuron> Neurons { get; }
    public int NeuronCount => Neurons?.Count ?? 0;
    public NeuronType Type;

    // в одном слое нейроны одного типа 
    public Layer(List<Neuron> neurons, NeuronType type = NeuronType.Normal)
    {
        Type = type;
        // проверить все входные нейроны на соответствие типу 
        Neurons = neurons;
    }
    
    // метод сбора выходных сигналов со всего слоя 
    public List<double> GetSignals()
    {
        var result = new List<double>();
        foreach (var neuron in Neurons)
        {
            result.Add(neuron.Output);
        }

        return result;
    }

    public override string ToString()
    {
        return Type.ToString();
    }
}