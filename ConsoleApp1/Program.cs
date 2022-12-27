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
        
        // Get data
        GetData data = new GetData();
        List<List<float>> X = new List<List<float>>();
        List<int> y = new List<int>();
        (X, y) = data.GetValues(path);

        List<List<float>> X_norm = data.NormalizeData(X);

        float[][] X_train = data.ToArrayCustom(X_norm);
        int[] y_train = y.ToArray();

        LogisticRegression process = new LogisticRegression();
        ultis func = new ultis();
        
        // Get shape
        (int m, int n) = data.GetShape(X_train);
        
        // Initialize weights
        var weights = func.GenerateRandomArray(n);
        
        // Compute cost
        var z = process.ComputeCost(X_train, y_train,weights,0);
        Console.WriteLine(z);
        
        //Compute gradient
        var (gradient_w, gradient_b) = process.CalculateGradient(y_train, X_train, weights, 0);
        //Console.WriteLine(gradient_b);

        (var w_in, var b_in, var J_his, var w_his) = process.GradientDescent(X_train, y_train, weights, 
            0.3f, 1000, 0.001f);
        /*foreach (var values in gradient_w)
        {
            Console.WriteLine(values);
        }*/

        /*float[][] test =
        {
            new float[] { 1.0f, 2.0f, 3.0f },
            new float[] { 4.0f, 5.0f, 6.0f },
            new float[] { 7.0f, 8.0f, 9.0f },
            new float[] { 10.0f, 11.0f, 12.0f }
        };
        var test_ = func.SliceArray2D(test, 0, 2, dimension: "two");
        func.Print2DArray(test_);*/


    }
}
