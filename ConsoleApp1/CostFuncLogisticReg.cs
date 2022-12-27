using System.Runtime.InteropServices.JavaScript;
using NumpyDotNet;
using System;
using System.IO;
using System.Linq;
using System.Numerics;
using nptorch = Numpy;
using Torch;
using npdotnet = NumpyDotNet.np;

namespace ConsoleApp1;

public class GetData
{
    //private string path = "E:/PhD BME admission test/ReviseKnowleged/software_in_c_sharp/Solution1/data.csv";
    //private string[] lines = File.ReadAllLines("E:/PhD BME admission test/ReviseKnowleged/software_in_c_sharp/Solution1/ConsoleApp1/data.csv");
    
    public (List<List<float>>, List<int>) GetValues(string path)
    {
        // Read csv
        string[] lines = File.ReadAllLines(path);
        
        // skip the first line which is the name of features in the csv file.
        var values = from line in lines.Skip(1)
            let parts = line.Split(',')
            select new
            {
                id = parts[0],
                
                // y ground truth
                diagnosis = parts[1],
                
                // X features
                radiusMean = parts[2],
                textureMean = parts[3],
                perimeterMean = parts[4],
                areaMean = parts[5],
                smoothnessMean = parts[6],
                compactnessMean = parts[7],
                concavityMean = parts[8],
                symmetryMean = parts[10],
                fractalDimensionMean = parts[11]
            };
        
        List<int> y = new List<int>();
        List<List<float>> X = new List<List<float>>();
        foreach (var value in values)
        {
            if (value.diagnosis == "M")
            {
                int a = 1;
                y.Add(a);
            }
            else if (value.diagnosis == "B")
            {
                int a = 0;
                y.Add(a);
            }
            else
            {
                y.Add(100);
            }
            
            float textureMean = (float)Convert.ToDouble(value.textureMean);
            float areaMean = (float)Convert.ToDouble(value.areaMean);
            float perimeterMean = (float)Convert.ToDouble(value.perimeterMean);
            float radiusMean = (float)Convert.ToDouble(value.radiusMean);
            float compactnessMean = (float)Convert.ToDouble(value.compactnessMean);
            float concavityMean = (float)Convert.ToDouble(value.concavityMean);
            float symmetryMean = (float)Convert.ToDouble(value.symmetryMean);
            
            List<float> features = new List<float>()
            {
                textureMean, areaMean, perimeterMean, radiusMean, compactnessMean,
                concavityMean, symmetryMean
            };
            X.Add(features);
        }
        
        /*ndarray X_train = np.array(X);
        ndarray y_train = np.array(y);*/
        return (X, y);
    }

    public List<List<float>> NormalizeData(List<List<float>> dataX)
    {
        // Find min and max
        float min = dataX.SelectMany(x => x).Min();
        float max = dataX.SelectMany(x => x).Max();
        
        // normalize
        List<List<float>> normalized = dataX.Select(x => 
            x.Select(y => 
                (y - min) / (max - min)).ToList()).ToList();
        
        return normalized;
    }

    public float[][] ToArrayCustom(List<List<float>> data)
    {
        var array = data.Select(list => list.ToArray()).ToArray();
        return array;
    }


    public float[][] Transpose(float[][] data)
    {
        float[][] transposed = Enumerable.Range(0, data[0].Length).Select(i => data.Select(row => row[i]).ToArray()).ToArray();
        return transposed;
    }
    
    
    public float[][] ToNumArray(List<List<float>> data)
    {
        //todo: the lib torch tensor doesnt work
        Vector<Vector<float>> vectorData = new Vector<Vector<float>>();
        /*for (int i = 0; i < data.Count; i++)
        {
            vectorData[i].Set();
        }*/
        //var flatArray = data.SelectMany(row => row).ToArray();
        var array = data.Select(list => list.ToArray()).ToArray();
        /*var num_array = nptorch.np.array(flatArray);
        var tensor = torch.tensor(num_array);
        
        var to_array = tensor.numpy();*/
        return array;
    }

    public (int, int) GetShape(float[][] data)
    {
        int m = data.Length;
        int n = data[0].Length;

        return (m, n);
    }
}


public class LogisticRegressionTest
{
    // define sigmoid function
    private double Sigmoid(float y)
    {
        return 1.0 / (1.0 + Math.Exp(-y));
    }
    
    // define loss function
    private float CostFuncLogistic(int y, float p)
    {
        /*
         * The loss needs to divide by number samples: m
         * loss = (1/m)*loss
         */
        if (p > 1.0)
        {
            throw new Exception("Probability is larger than 1");
        }
        float loss = -y * MathF.Log(p) - (1 - y) * MathF.Log(1 - p);
        return loss;
    }
    
    public float ComputeCost(float[][] X, int[] y, float[] weights, float bias)
    {
        /*
         * Computes the cost over all examples
            Args:
              X : (ndarray Shape (m,n)) data, m examples by n features
              y : (array_like Shape (m,)) target value 
              w : (array_like Shape (n,)) Values of parameters of the model      
              b : scalar Values of bias parameter of the model
              lambda_: unused placeholder
            Returns:
              total_cost: (scalar)         cost 
         */
        // calculate w * x
        float Cost = 0;
        
        for (int i = 0; i < X.Length; i++)
        {
            float[] dotProduct = new float[X[i].Length];
            float f_wb = 0;
            for (int j = 0; j < X[i].Length; j++)
            {
                dotProduct[j] = (float)Sigmoid((X[i][j] * weights[j] + bias));
                f_wb = dotProduct[j];
            }

            Cost += CostFuncLogistic(y[i], f_wb);
        }
        
        float TotalCost = Cost / X.Length;
        
        return TotalCost;
    }
    
    private float Predict(float[] x, float[] w, float b)
    {
        float p = b; // predicted output initialized to the bias
        for (int i = 0; i < x.Length; i++)
        {
            p += x[i] * w[i]; // add the product of the feature value and the weight to the predicted output
        }
        return p + b;
    }
    
    /// <summary>
    /// Computes the gradient for logistic regression 
    /// </summary>
    /// <param name="targets"> (array_like Shape (m,1)) actual value </param> 
    /// <param name="X">(ndarray Shape (m,n)) variable such as house size </param> 
    /// <param name="w">(array_like Shape (n,1)) values of parameters of the model </param>  
    /// <param name="b">(scalar)                 value of parameter of the model </param> 
    /// <returns>
    /// dj_dw: (array_like Shape (n,1)) The gradient of the cost w.r.t. the parameters w.
    /// dj_db: (scalar)                The gradient
    /// </returns>
    public (float[], float) CalculateGradient(int[] targets, float[][] X, float[] w, float b)
    {
        GetData data = new GetData();
        (int m, int n) = data.GetShape(X);
        
        float[] gradient_w = new float[n];
        float gradient_b = 0.0f;
        
        for (int i = 0; i < m; i++)
        {
            float[] x = X[i];
            int y = targets[i];
            float p = Predict(x, w, b); // predicted output for the current sample
            float error = p - y; // error between the predicted output and the target
            for (int j = 0; j < n; j++)
            {
                gradient_w[j] += error * x[j]; // update gradient with respect to weight j
            }
            gradient_b += error; // update gradient with respect to bias
        }

        // divide the gradients by the number of samples
        for (int j = 0; j < n; j++)
        {
            gradient_w[j] /= m;
        }
        gradient_b /= m;

        return (gradient_w, gradient_b);
    }

    public (float[], float, List<float>, List<float>) GradientDescent(float[][] X, int[] y, float[] weights, 
        float bias, int NumIters, float alpha)
    {
        /*
         * Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
        Args:
          X :    (array_like Shape (m, n)
          y :    (array_like Shape (m,))
          w_in : (array_like Shape (n,))  Initial values of parameters of the model
          b_in : (scalar)                 Initial value of parameter of the model
          cost_function:                  function to compute cost
          alpha : (float)                 Learning rate
          num_iters : (int)               number of iterations to run gradient descent
          lambda_ (scalar, float)         regularization constant
          
        Returns:
          w : (array_like Shape (n,)) Updated values of parameters of the model after
              running gradient descent
          b : (scalar)                Updated value of parameter of the model after
              running gradient descent
         */
        // Get number of samples and features
        GetData data = new GetData();
        (int m, int n) = data.GetShape(X);
        
        List<float> J_history = new List<float>();
        List<float> W_history = new List<float>();

        float[] w_in = new float[n];
        float b_in = 0.0f;
        for (int i = 0; i < NumIters; i++)
        {
            (float[] dj_dw, float dj_db) = CalculateGradient(y, X, weights, bias);
            
            // update parameters
            float w_in_add = 0.0f;
            for (int j = 0; j < dj_dw.Length; j++)
            {
                w_in[j] -= alpha * dj_dw[j];
                w_in_add = w_in[j];
            }

            b_in -= alpha * dj_db;
            
            // Save cost J at each iterations
            float print_cost = 0.0f;
            if (i < NumIters)
            {
                float cost = ComputeCost(X, y, w_in, b_in);
                J_history.Add(cost);
                print_cost = cost;
            }

            if (i % 10 == 0)
            {
                W_history.Add(w_in_add);
                Console.WriteLine($"Iteration {i}: Cost = {print_cost}");
            }
        }

        return (w_in, b_in, J_history, W_history);
    }
}

public class LogisticRegression
    {
        private readonly float[] coefficients;
        private readonly float intercept;

        public LogisticRegression(float[][] X_train, int[] y_train, int numIterations = 1000, float learningRate = 0.01f)
        {
            // Initialize coefficients and intercept
            int numFeatures = X_train[0].Length;
            coefficients = new float[numFeatures];
            intercept = 0;

            // Perform gradient descent to optimize coefficients and intercept
            for (int i = 0; i < numIterations; i++)
            {
                float[] gradientCoefficients = new float[numFeatures];
                float gradientIntercept = 0;
                List<float> print_error = new List<float>();
                for (int j = 0; j < X_train.Length; j++)
                {
                    float predictedProbability = PredictProbability(X_train[j]);
                    float error = y_train[j] - predictedProbability;
                    
                    // add error to print
                    print_error.Add(error);
                    for (int k = 0; k < numFeatures; k++)
                    {
                        gradientCoefficients[k] += error * X_train[j][k];
                    }
                    gradientIntercept += error;
                }
                for (int j = 0; j < numFeatures; j++)
                {
                    coefficients[j] += learningRate * gradientCoefficients[j] / X_train.Length;
                }
                intercept += learningRate * gradientIntercept / X_train.Length;

                float mean_error = print_error.Average();
                Console.WriteLine($"Iterations {i}   Loss = {-mean_error}");
            }
        }

        public float PredictProbability(float[] x)
        {
            // Calculate predicted probability using coefficients and intercept
            float sum = intercept;
            for (int i = 0; i < x.Length; i++)
            {
                sum += x[i] * coefficients[i];
            }
            return (float)(1 / (1 + Math.Exp(-sum)));
        }

        public int Predict(float[] x)
        {
            // Predict class label based on predicted probability
            return PredictProbability(x) > 0.5 ? 1 : 0;
        }
    }
    
public static class Evaluation
{
    public static float CalculateAccuracy(LogisticRegression model, float[][] X_test, int[] y_test)
    {
        // Make predictions for test data and calculate accuracy
        int numCorrect = 0;
        for (int i = 0; i < X_test.Length; i++)
        {
            if (model.Predict(X_test[i]) == y_test[i])
            {
                numCorrect++;
            }
        }
        Console.WriteLine($"Accuracy of model = {(float)numCorrect / X_test.Length}");
        return (float)numCorrect / X_test.Length;
    }
}