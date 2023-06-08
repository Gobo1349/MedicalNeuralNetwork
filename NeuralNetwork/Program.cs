// See https://aka.ms/new-console-template for more information

using System.Diagnostics;
using NeuralNetwork;
using Xunit;
using NUnit.Framework;
using Assert = NUnit.Framework.Assert;

/*
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
    { 1, 1, 0, 0 },
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
    var res = neuralNetwork.FeedForward(row).Output;
    results.Add(res);
}

// проверка 
for (int i = 0; i < results.Count; i++)
{
    var expected = Math.Round(outputs[i], 2);
    var actual = Math.Round(results[i], 2);
    Assert.AreEqual(expected, actual);
} */
// создаем нейроны сх, вых, промежут слоев 

// теперь задаем коэффициенты веса 
// 0 слой - все коэффициенты 1 
//neuralNetwork.Layers[1].Neurons[0].SetWeights(0.5, -0.1, 0.3, -0.1);
//neuralNetwork.Layers[1].Neurons[1].SetWeights(0.1, -0.3, 0.7, -0.3);
//neuralNetwork.Layers[2].Neurons[0].SetWeights(1.2, 0.8);

// подаем на вход сигнал 
// double[] nums = { 1, 0, 0, 0 };
// var result = neuralNetwork.FeedForward(nums);

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
        var input = values.Take(values.Count - 1)
            .ToArray(); // последний элемент - результат, он не нужен во входных данных 

        outputs.Add(output);
        inputs.Add(input);
    }
}



var inputSignals = new double[inputs.Count, inputs[0].Length]; // преобразование входных данных к двумерному массиву 
for (int i = 0; i < inputSignals.GetLength(0); i++)
{
    for (int k = 0; k < inputSignals.GetLength(1); k++)
    {
        inputSignals[i, k] = inputs[i][k];
    }
}

// обучение
var topology = new Topology(outputs.Count, 1, 0.1, outputs.Count / 2); // топология предположительной сети - вх вых и скрытые слои 
var neuralNetwork = new NeuralNetwork.NeuralNetwork(topology); // создаем саму сеть 
var difference = neuralNetwork.Learn(outputs.ToArray(), inputSignals, 1000); // среднеквадратич ошибка 


// непосредственно работа
var results = new List<double>();
for (int i = 0; i < outputs.Count; i++)
{
    var res = neuralNetwork.Predict(inputs[i]).Output;
    results.Add(res);
}

// проверка 
for (int i = 0; i < results.Count; i++)
{
    var expected = Math.Round(outputs[i], 2);
    var actual = Math.Round(results[i], 2);
    Assert.AreEqual(expected, actual);
}

