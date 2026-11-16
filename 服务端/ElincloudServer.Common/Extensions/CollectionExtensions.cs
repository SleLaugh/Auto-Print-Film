
public static class CollectionExtensions
{
    public static List<T> Distinct<T, V>(this IEnumerable<T> source, Func<T, V> keySelector)
    {
        return source.Distinct(new DistinctComparer<T, V>(keySelector)).ToList();
    }
}

public class DistinctComparer<T, V> : IEqualityComparer<T>
{
    private Func<T, V> keySelector;

    public DistinctComparer(Func<T, V> keySelector)
    {
        this.keySelector = keySelector;
    }

    public bool Equals(T x, T y)
    {
        return EqualityComparer<V>.Default.Equals(keySelector(x), keySelector(y));
    }

    public int GetHashCode(T obj)
    {
        return EqualityComparer<V>.Default.GetHashCode(keySelector(obj));
    }
}