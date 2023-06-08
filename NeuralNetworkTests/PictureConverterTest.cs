using System.Diagnostics;
using NeuralNetwork;
using Xunit;
using NUnit.Framework;
using Assert = NUnit.Framework.Assert;

namespace NeuralNetworkTests;

public class PictureConverterTest
{
    [Test]
    public void ConverterTest()
    {
        var converter = new PictureConverter();
        var inputs = converter.Convert(@"C:\Users\misamsonov\RiderProjects\NeuralNetwork\NeuralNetwork\NeuralNetworkTests\Images\Parazited.png");
        converter.Save(@"C:\Users\misamsonov\RiderProjects\NeuralNetwork\NeuralNetwork\image.png", inputs);
    }
    
   
}