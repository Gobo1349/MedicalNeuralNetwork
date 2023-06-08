namespace NeuralNetwork;

public class Neuron
{
    public List<double> Weights { get;  } // веса связей 
    public List<double> Inputs { get;  } // входные сигналы 

    public NeuronType NeuronType { get; }
    public double Output { get; private set; } // результат
    public double Delta { get; private set; } // переменная для промежуточных вычислений 


    public Neuron(int inputCount, NeuronType type = NeuronType.Normal) // кол-во входных связей - тип нейрона 
    {
        NeuronType = type;
        // по количеству вх сигналов мы можем определить количество весов (один вес на каждый сигнал)

        Weights = new List<double>();
        Inputs = new List<double>();
        InitWeightsRandomValue(inputCount);
    }

    private void InitWeightsRandomValue(int inputCount)
    {
        var rnd = new Random();
        for (int i = 0; i < inputCount; i++)
        {
            if (NeuronType == NeuronType.Input) // у вх нейронов вес всегда 1 
            {
                Weights.Add(1);
            }
            else
            {
                Weights.Add(rnd.NextDouble()); // случайное значение веса от 0 до 1    
            }
            Inputs.Add(0);
        }
    }

    // вычисление 
    public double FeedForward(List<double> inputs) // движение слева направо - на вход - входные значения
    {
        for (int i = 0; i < inputs.Count; i++) // сохранение вх сигналов 
        {
            Inputs[i] = inputs[i];
        }
        var sum = 0.0;
        for (int i = 0; i < inputs.Count; i++)
        {
            sum += inputs[i] * Weights[i];
        }

        if (NeuronType != NeuronType.Input) // применяем сигмоиду на всех нейронах, кроме входных 
        {
            Output = Sigmoid(sum);
        }
        else
        {
            Output = sum; 
        }
        return Output;
    }
    
    // мат функция - сигмоида
    private double Sigmoid(double x)
    {
        var result = 1.0 / (1.0 + Math.Pow(Math.E, -x));
        return result; 
    }
    
    // производная от сигмоиды 
    private double SigmoidDx(double x)
    {
        var sigmoid = Sigmoid(x);
        var result = sigmoid / (1 - sigmoid);
        return result;
    }

    public void Learn(double error, double learningRate) // вычисление новых весов 
    {
        if (NeuronType == NeuronType.Input) // входные нейроны не обучаются 
        {
            return;
        }

        Delta = error * SigmoidDx(Output);
        for (int i = 0; i < Weights.Count; i++)
        {
            var weight = Weights[i];
            var input = Inputs[i];
            
            // вычисляем новый вес 
            var newWeight = weight - input * Delta * learningRate;
            Weights[i] = newWeight;
        }
    }
    /*public void SetWeights(params double[] weights)
    {
        // удалить после добавления возможности обучения сети 
        for (int i = 0; i < weights.Length; i++)
        {
            Weights[i] = weights[i];
        }
    }*/

    public override string ToString() // для отладки 
    {
        return Output.ToString();
    }
}