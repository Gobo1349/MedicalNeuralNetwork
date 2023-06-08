namespace NeuralNetwork;

public class NeuralNetwork // нейронная сеть - набор слоев 
{
    public Topology Topology { get; }
    public List<Layer> Layers { get; }

    public NeuralNetwork(Topology topology)
    {
        Topology = topology;
        Layers = new List<Layer>();
        
        // заполнение слоев методами 
        CreateInputLayer();
        CreateHiddenLayers();
        CreateOutputLayer();
    }

    public Neuron Predict(params double[] inputSignals)
    {
        SendSignalsToInputNeurons(inputSignals); // отправка данных на вход 
        FeedForwardAllNeuronsAfterInput(); // ??собираем все сигналы с предыдущего слоя для обработки на след слое 
        
        // конечный результат
        if (Topology.OutPutCount == 1) // если на выходе один нейрон, то мы возвращаем значение этого нейрона 
        {
            return Layers.Last().Neurons[0];
        }
        else // если вых нейронов несколько 
        {
            return Layers.Last().Neurons.OrderByDescending(n => n.Output).First(); // нейрон с самым большим выходным значением 
        }
    }

    public double Learn(double[] expected, double[,] inputs, int era) // ожидаемый результат - вх данные - кол-во эпох - сколько раз прогоняем обучение
    {
        var signals = Normalization(inputs);
        var error = 0.0;
        for (int i = 0; i < era; i++)
        {
            for (int k = 0; k < expected.Length; k++)
            {
                var output = expected[k];
                var input = GetRow(signals, k);
                
                error += BackPropagation(output, input);

            }
        }

        var result = error / era; // средняя ошибка
        return result;
    }
    
    public static double[] GetRow(double[,] matrix, int row) // получить одну строку двумерного массива
    {
        var columns = matrix.GetLength(1);
        var array = new double[columns];
        for (int i = 0; i < columns; ++i)
            array[i] = matrix[row, i];
        return array;
    }
    
    // метод обр распространения ошибки
    private double BackPropagation(double expected, params double[] inputs)
    {
        var actual = Predict(inputs).Output; // реальный результат
        
        // ошибка для выходного слоя 
        var difference = actual - expected;

        foreach (var neuron in Layers.Last().Neurons)
        {
            neuron.Learn(difference, Topology.LearningRate); // обучаем выходной нейрон 
        }

        for (int k = Layers.Count - 2; k >= 0; k--)
        {
            var layer = Layers[k];
            var previousLayer = Layers[k + 1]; 
            
            // обучаем нейроны внутри одного слоя 
            for (int i = 0; i < layer.NeuronCount; i++) // перебор нейронов на промежуточном слое 
            {
                var neuron = layer.Neurons[i];

                for (int q = 0; q < previousLayer.NeuronCount; q++) // перебор связей справа 
                {
                    var previousNeuron = previousLayer.Neurons[q];
                    var error = previousNeuron.Weights[i] * previousNeuron.Delta; // первый нейрон - первый вес, второй нейрон - второй вес и т д 
                    
                    // обучаем текущий нейрон 
                    neuron.Learn(error, Topology.LearningRate);
                }
            }
        }

        var result = difference * difference; // квадратичная - чтобы было заметнее
        return result; 
    }

    private void FeedForwardAllNeuronsAfterInput()
    {
        for (int i = 1; i < Layers.Count; i++) // первый слой уже обработали 
        {
            var layer = Layers[i];
            var previousLayerSignals = Layers[i - 1].GetSignals();

            // перебираем все нейроны этого слоя и отправляем туда все сигналы с предыдущего слоя 
            foreach (var neuron in layer.Neurons)
            {
                neuron.FeedForward(previousLayerSignals);
            }
        }
    }

    private void SendSignalsToInputNeurons(params double[] inputSignals)
    {
        for (int i = 0; i < inputSignals.Length; i++)
        {
            var signal = new List<double> { inputSignals[i] }; // коллекция из одного элемента 
            var neuron = Layers[0].Neurons[i];

            neuron.FeedForward(signal); // отправка первоначальных данных в сеть 
        }
    }

    private void CreateOutputLayer()
    {
        var outputNeurons = new List<Neuron>(); // коллекция выходных нейронов 
        // связи между нейронами - каждый - каждый послойно 
        var lastLayer = Layers.Last(); // для определения количества входов у нейронов на выходном слое 
        for (int i = 0; i < Topology.OutPutCount; i++)
        {
            var neuron = new Neuron(lastLayer.NeuronCount, NeuronType.Output); // у входных нейронов всегда один вход 
            outputNeurons.Add(neuron);
        }

        var outputLayer = new Layer(outputNeurons, NeuronType.Output);
        Layers.Add(outputLayer);
    }

    private void CreateHiddenLayers() // промежуточные слои 
    {
        for (int k = 0; k < Topology.HiddenLayers.Count; k++)
        {
            var hiddenNeurons = new List<Neuron>(); 
            var lastLayer = Layers.Last(); 
            for (int i = 0; i < Topology.HiddenLayers[k]; i++)
            {
                var neuron = new Neuron(lastLayer.NeuronCount); // тип - normal по умолчанию 
                hiddenNeurons.Add(neuron);
            }

            var hiddenLayer = new Layer(hiddenNeurons);
            Layers.Add(hiddenLayer);
        }
    }

    private void CreateInputLayer()
    {
        var inputNeurons = new List<Neuron>(); // коллекция входных нейронов 
        for (int i = 0; i < Topology.InputCount; i++)
        {
            var neuron = new Neuron(1, NeuronType.Input); // у входных нейронов всегда один вход 
            inputNeurons.Add(neuron);
        }

        var inputLayer = new Layer(inputNeurons, NeuronType.Input);
        Layers.Add(inputLayer);
    }
    
    //  Нормализация и масштабирование данных
    
    // метод масштабирования 
    private double[,] Scalling(double[,] inputs) // таблица вх данных 
    {
        var result = new double[inputs.GetLength(0), inputs.GetLength(1)];
        
        for (int column = 0; column < inputs.GetLength(1); column++) // считаем по колонкам
        {
            // ищем минимум и максимум каждого столбца
            var min = inputs[0, column];
            var max = inputs[0, column];

            for (int row = 1; row < inputs.GetLength(0); row++)
            {
                var item = inputs[row, column];

                if (item < min)
                {
                    min = item;
                }

                if (item > max)
                {
                    max = item; 
                }
            }

            var diff = max - min;
            // нормализуем значения по формуле 
            for (int row = 1; row < inputs.GetLength(0); row++)
            {
                result[row, column] = (inputs[row, column] - min) / diff;
            }
        }

        return result;
    }
    
    // метод нормализации 
    private double[,] Normalization(double[,] inputs) // таблица вх данных 
    {
        var result = new double[inputs.GetLength(0), inputs.GetLength(1)];
        
        for (int column = 0; column < inputs.GetLength(1); column++) // считаем по колоннам
        {
            // вычисляем среднее знач столбца - среднее знач сигнала нейрона 
            var sum = 0.0; 
            for (int row = 0; row < inputs.GetLength(0); row++)
            {
                sum += inputs[row, column];
            }
            var average = sum / inputs.GetLength(0);
            
            // стандартное квадратичное отклонение нейрона 
            var error = 0.0; 
            for (int row = 0; row < inputs.GetLength(0); row++)
            {
                error += Math.Pow((inputs[row, column] - average), 2);
            }
            var standardError = Math.Sqrt(error / inputs.GetLength(0));
            
            for (int row = 0; row < inputs.GetLength(0); row++)
            {
                result[row, column] = (inputs[row, column] - average) / standardError;
            }
        }

        return result;
    }
}
