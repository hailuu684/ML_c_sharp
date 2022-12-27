using System.Runtime.Intrinsics;

namespace ConsoleApp1;

public class DotProduct
{
    public double[] Vector1 { get; }
    public double[] Vector2 { get; }
    
    public DotProduct(double[] vector1, double[] vector2)
    {
        this.Vector1 = vector1;
        this.Vector2 = vector2;
    }

    public double Calculate()
    {
        double dotProduct = 0;
            
        for (int i = 0; i < this.Vector1.Length; i++)
        {
            dotProduct += this.Vector1[i] * this.Vector2[i];
        }

        return dotProduct;
    }
}