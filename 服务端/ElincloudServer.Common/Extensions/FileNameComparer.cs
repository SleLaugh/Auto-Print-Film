
namespace ElincloudServer.Common
{
    ///<summary>
    ///比较两个字符串，如果含用数字，则数字按数字的大小来比较。
    ///</summary>
    public class FileNameComparer : IComparer<string>
    {
        // Calls CaseInsensitiveComparer.Compare with the parameters reversed.
        ///<summary>
        ///比较两个字符串，如果含用数字，则数字按数字的大小来比较。
        ///</summary>
        ///<param name="x"></param>
        ///<param name="y"></param>
        ///<returns></returns>
        int IComparer<string>.Compare(string x, string y)
        {
            if (x == null || y == null)
                throw new ArgumentException("Parameters can't be null");
            char[] arr1 = x.ToCharArray();
            char[] arr2 = y.ToCharArray();
            int i = 0, j = 0;
            while (i < arr1.Length && j < arr2.Length)
            {
                if (char.IsDigit(arr1[i]) && char.IsDigit(arr2[j]))
                {
                    string s1 = "", s2 = "";
                    decimal d1 = 0, d2 = 0;
                    while (i < arr1.Length && char.IsDigit(arr1[i]))
                    {
                        s1 += arr1[i];
                        i++;
                    }
                    while (j < arr2.Length && char.IsDigit(arr2[j]))
                    {
                        s2 += arr2[j];
                        j++;
                    }

                    if (decimal.TryParse(s1, out d1) && decimal.TryParse(s2, out d2))
                    {
                        if (d1 > d2)
                        {
                            return 1;
                        }
                        if (d1 < d2)
                        {
                            return -1;
                        }
                    }
                    else
                    {
                        int v = s1.CompareTo(s2);
                        if (v != 0)
                        {
                            return v;
                        }
                    }
                }
                else
                {
                    if (arr1[i] > arr2[j])
                    {
                        return 1;
                    }
                    if (arr1[i] < arr2[j])
                    {
                        return -1;
                    }
                    i++;
                    j++;
                }
            }
            if (arr1.Length == arr2.Length)
            {
                return 0;
            }
            else
            {
                return arr1.Length > arr2.Length ? 1 : -1;
            }
        }
    }
}
