// See https://aka.ms/new-console-template for more information

using System.Diagnostics;
using NeuralNetwork;
using Xunit;
using NUnit.Framework;
using Assert = NUnit.Framework.Assert;

public class Tests
{
    [Test]
    public void FeedForwardTest()
    {
        var outputs = new double[] { 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1 };
        var inputs = new double[,]
        {
            // Результат - Пациент болен - 1
            //             Пациент Здоров - 0
        
            // Неправильная температура T
            // Хороший возраст A
            // Курит S
            // Правильно питается F
            //T  A  S  F
            { 0, 0, 0, 0 },
            { 0, 0, 0, 1 },
            { 0, 0, 1, 0 },
            { 0, 0, 1, 1 },
            { 0, 1, 0, 0 },
            { 0, 1, 0, 1 },
            { 0, 1, 1, 0 },
            { 0, 1, 1, 1 },
            { 1, 0, 0, 0 },
            { 1, 0, 0, 1 },
            { 1, 0, 1, 0 },
            { 1, 0, 1, 1 },
            { 1, 1, 0, 0 }
            { 1, 1, 0, 1 },
            { 1, 1, 1, 0 },
            { 1, 1, 1, 1 }
        };
        
        // обучение
        var topology = new Topology(4, 1, 0.1, 2); // топология предположительной сети - вх вых и скрытые слои 
        var neuralNetwork = new NeuralNetwork.NeuralNetwork(topology); // создаем саму сеть 
        var difference = neuralNetwork.Learn(outputs, inputs, 10000); // среднеквадратич ошибка 
        
        // непосредственно работа
        var results = new List<double>();
        for (int i = 0; i < outputs.Length; i++)
        {
            var row = NeuralNetwork.NeuralNetwork.GetRow(inputs, i);
            var res = neuralNetwork.Predict(row).Output;
            results.Add(res);
        }
        
        // проверка 
        for (int i = 0; i < results.Count; i++)
        {
            var expected = Math.Round(outputs[i], 2);
            var actual = Math.Round(results[i], 2);
            Assert.AreEqual(expected, actual);
        }
    }

    [Test]
    public void DatasetTest()
    {
        var outputs = new List<double>();
        var inputs = new List<double[]>();
        using (var sr = new StreamReader("heart.csv"))
        {
            var header = sr.ReadLine();

            while (!sr.EndOfStream)
            {
                var row = sr.ReadLine();
                var temp = row.Split(',');
                var values = temp.Select(v => Convert.ToDouble(v.Replace(".", ","))).ToList();
                var output = values.Last();
                var input = values.Take(values.Count - 1).ToArray(); // последний элемент - результат, он не нужен во входных данных 
                
                outputs.Add(output);
                inputs.Add(input);
            }
        }
    }
    
     [Test]
    public void RecognizeImages()
    {
        var parasitizedPath = @"C:\Users\misamsonov\RiderProjects\NeuralNetwork\NeuralNetwork\Parasitized\";
        var unparasitizedPath = @"C:\Users\misamsonov\RiderProjects\NeuralNetwork\NeuralNetwork\Uninfected\";

        var converter = new PictureConverter();
        var testParasitizedImageInput = converter.Convert(@"C:\Users\misamsonov\RiderProjects\NeuralNetwork\NeuralNetwork\NeuralNetworkTests\Images\Parazited.png");
        var testUnparasitizedImageInput = converter.Convert(@"C:\Users\misamsonov\RiderProjects\NeuralNetwork\NeuralNetwork\NeuralNetworkTests\Images\Unparazited.png");

        var topology = new Topology(testParasitizedImageInput.Count, 1, 0.1, testParasitizedImageInput.Count / 2);
        var neuralNetwork = new NeuralNetwork.NeuralNetwork(topology);

        var parasitizedInputs = GetData(parasitizedPath, testParasitizedImageInput, converter, 1000);
        neuralNetwork.Learn(new double[] { 1 }, parasitizedInputs, 1); // обучаем сеть на зараженных клетках 
        
        var unparasitizedInputs = GetData(unparasitizedPath, testParasitizedImageInput, converter, 1000);
        neuralNetwork.Learn(new double[] { 0 }, unparasitizedInputs, 1); // обучаем сеть на здоровых клетках

        var par = neuralNetwork.Predict(testParasitizedImageInput.Select(t => (double)t).ToArray());
        var unpar = neuralNetwork.Predict(testUnparasitizedImageInput.Select(t => (double)t).ToArray());
        
        Assert.AreEqual(1, Math.Round(par.Output, 2));
        Assert.AreEqual(0, Math.Round(unpar.Output, 2));


    }

    private static double[,] GetData(string parasitizedPath, List<int> testImageInput, PictureConverter converter, int size) // формируем вх данные из картинок для обучения сети
    {
        var images = Directory.GetFiles(parasitizedPath);
        var result = new double[size, testImageInput.Count];

        for (int i = 0; i < size; i++)
        {
            var image = converter.Convert(images[i]);
            for (int k = 0; k < image.Count; k++)
            {
                result[i, k] = image[k]; // заполняем массив для обучения 
            }
        }

        return result;
    }
}
// создаем нейроны сх, вых, промежут слоев 

// теперь задаем коэффициенты веса 
// 0 слой - все коэффициенты 1 
//neuralNetwork.Layers[1].Neurons[0].SetWeights(0.5, -0.1, 0.3, -0.1);
//neuralNetwork.Layers[1].Neurons[1].SetWeights(0.1, -0.3, 0.7, -0.3);
//neuralNetwork.Layers[2].Neurons[0].SetWeights(1.2, 0.8);

// подаем на вход сигнал 
// double[] nums = { 1, 0, 0, 0 };
// var result = neuralNetwork.FeedForward(nums);