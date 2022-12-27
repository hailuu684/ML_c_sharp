namespace ConsoleApp1;

public delegate int Transformer(int x);

public class Delegate
{
    public static int Square(int a)
    {
        return a * a;
    }

    public static int Cube(int a)
    {
        return a * a * a;
    }

    public List<int> Transform(List<int> list, Transformer t)
    {
        var result = new List<int>();
        foreach (int x in list)
        {
            result.Add(t(x));
        }

        return result;
    }
    
}