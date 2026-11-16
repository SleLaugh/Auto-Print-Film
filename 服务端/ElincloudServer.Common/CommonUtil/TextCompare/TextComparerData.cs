
namespace ElincloudServer.Common
{
    internal class TextComparerData
    {

        /// <summary>Number of elements (lines).</summary>
        internal int Length;

        /// <summary>Buffer of numbers that will be compared.</summary>
        internal int[] data;

        /// <summary>
        /// Array of booleans that flag for modified data.
        /// This is the result of the diff.
        /// This means deletedA in the first Data or inserted in the second Data.
        /// </summary>
        internal bool[] modified;

        /// <summary>
        /// Initialize the Diff-Data buffer.
        /// </summary>
        /// <param name="data">reference to the buffer</param>
        internal TextComparerData(int[] initData)
        {
            data = initData;
            Length = initData.Length;
            modified = new bool[Length + 2];
        }

    }
}
