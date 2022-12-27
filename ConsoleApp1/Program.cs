// See https://aka.ms/new-console-template for more information
using System;
using ConsoleApp1;
using Numpy.Models;
using NumpyDotNet;
using Torch;
using Delegate = ConsoleApp1.Delegate;


public class Program
{
    public static void Main()
    {   
        // Delegate examples, uncommnent to run 
        /*
        Transformer del = Delegate.Square;
        int square = del(10);
        
        Console.WriteLine(square);
        
        del = Delegate.Cube;
        int cube = del(12);
        Console.WriteLine(cube);
        */
        
        // Property examples, uncommnent to run 
        /*Property person = new Property();
        person.Name = "John";
        Console.WriteLine(person.Name); // "John"*/
        
        // Dot Product
        // Create two vectors of length 3
        /*double[] vector1 = {1, 2, 3};
        double[] vector2 = {4, 5, 6};
        var initialVector = new DotProduct(vector1, vector2);
        var result = initialVector.Calculate();
        Console.WriteLine("result ="+ result);*/
        
        string path = "E:/PhD BME admission test/ReviseKnowleged/software_in_c_sharp/Solution1/ConsoleApp1/data.csv";
        ultis func = new ultis();
        
        // Get data
        GetData data = new GetData();
        List<List<float>> X = new List<List<float>>();
        List<int> y = new List<int>();
        (X, y) = data.GetValues(path);
        
        // Normalize data with standard deviation and mean
        List<List<float>> X_norm = data.NormalizeData(X);
        
        // convert list<list<float>> to float[][]
        float[][] array = X.Select(x => x.ToArray()).ToArray();
        
        // Normalize data using StandardScaler
        StandardScaler scaler = new StandardScaler(array);
        float[][] X_scaled = scaler.Transform(array);
        
        // Convert data to Array
        //float[][] X_train = data.ToArrayCustom(X_norm);
        int[] y_converted = y.ToArray();
        
        // split data
        int start = 0;
        int end = (int)Math.Round(X_scaled.Length * 0.8);
        float[][] X_train = func.SliceArray2D(X_scaled, start, end, "one");
        float[][] X_test = func.SliceArray2D(X_scaled, end, X_scaled.Length, "one");
        int[] y_train = func.SliceArrayInt(y_converted, start, end);
        int[] y_test = func.SliceArrayInt(y_converted, end, y_converted.Length);
        
        /*// get number of M and B
        int ones = 0;
        int zeros = 0;

        foreach (int element in y_train)
        {
            if (element == 1)
            {
                ones++;
            }
            else if (element == 0)
            {
                zeros++;
            }
        }

        Console.WriteLine("Number of Malinan: " + ones);
        Console.WriteLine("Number of zeros: " + zeros);*/
        //Number of M: 212
        //Number of B: 357
        
        LogisticRegression model = new LogisticRegression(X_train, y_train, 1200);
        float accuracy = Evaluation.CalculateAccuracy(model, X_test, y_test);
        

    }
}
