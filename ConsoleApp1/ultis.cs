using Python.Runtime;

namespace ConsoleApp1;

public class ultis
{
    public float[] GenerateRandomArray(int size)
    {
        var random = new Random();
        var values = new float[size];
        for (int i = 0; i < size; i++)
        {
            values[i] = (float)random.NextDouble();
        }

        return values;
    }
    
    public double[][] ConvertToArray2D(double[] array, int m, int n)
    {
        double[][] array2d = new double[m][];
        for (int i = 0; i < m; i++)
        {
            array2d[i] = new double[n];
            for (int j = 0; j < n; j++)
            {
                array2d[i][j] = array[i * n + j];
            }
        }

        return array2d;
    }

    public void PrintArray1D(double[] array)
    {
        if (array.Length > 1)
        {
            foreach (var value in array)
            {
                Console.WriteLine(value);
            }
        }
        else
        {
            Console.Write(array);
        }
    }

    public void Print2DArray(float[][] array)
    {
        for (int i = 0; i < array.Length; i++)
        {
            for (int j = 0; j < array[i].Length; j++)
            {
                Console.Write(array[i][j] + " ");
            }
            Console.WriteLine();
        }
    }
    /// <summary>
    /// Slice the array of dtype float[]
    /// </summary>
    /// <param name="array">Array with dtype float[]</param>
    /// <param name="start">start of slice</param>
    /// <param name="end">end of slice</param>
    /// <returns>sliced array</returns>
    public float[] SliceArrayFloats(float[] array, int start, int end)
    {
        return array.Skip(start).Take(end - start).ToArray();
    }

    public int[] SliceArrayInt(int[] array, int start, int end)
    {
        return array.Skip(start).Take(end - start).ToArray();
    }
    /// <summary>
    /// Slice the array of dtype float[][]
    /// use dimension="one" to slice the first dimension
    /// use dimension="towo" to slice the second dimension
    /// </summary>
    /// <param name="array"></param>
    /// <param name="start"></param>
    /// <param name="end"></param>
    /// <param name="dimension">use "one" or "two"</param>
    /// <returns></returns>
    public float[][] SliceArray2D(float[][] array, int start, int end,
        string dimension)
    {
        if (dimension == "one")
        {
            float[][] result = new float[end - start][];
            Array.Copy(array, start, result, 0, end - start);
            return result;
        }

        if (dimension == "two")
        {
            float[][] result = new float[array.Length][];
            for (int i = 0; i < array.Length; i++)
            {
                result[i] = new float[end - start];
                Array.Copy(array[i], start, 
                    result[i], 0, 
                    end - start);
            }

            return result;
        }
        
        throw new ArgumentException("Invalid argument dimension");
        //return arrayFail;
    }
}

public class StandardScaler
{
    private readonly float[] means;
    private readonly float[] stddevs;

    public StandardScaler(float[][] data)
    {
        // Calculate means and standard deviations for each feature
        int numFeatures = data[0].Length;
        means = new float[numFeatures];
        stddevs = new float[numFeatures];
        for (int i = 0; i < numFeatures; i++)
        {
            float[] feature = data.Select(x => x[i]).ToArray();
            means[i] = feature.Average();
            stddevs[i] = MathF.Sqrt(feature.Select(x => MathF.Pow(x - means[i], 2)).Sum() / (feature.Length - 1));
        }
    }

    public float[][] Transform(float[][] data)
    {
        // Standardize the features of the data
        int numFeatures = data[0].Length;
        float[][] standardized = new float[data.Length][];
        for (int i = 0; i < data.Length; i++)
        {
            standardized[i] = new float[numFeatures];
            for (int j = 0; j < numFeatures; j++)
            {
                standardized[i][j] = (data[i][j] - means[j]) / stddevs[j];
            }
        }
        return standardized;
    }
}