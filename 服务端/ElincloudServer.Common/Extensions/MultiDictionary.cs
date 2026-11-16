using System.Collections;

namespace ElincloudServer.Common
{
    public class MultiDictionary<TKey, TValue> : IDictionary<TKey, TValue>
    {
        private Dictionary<TKey, List<TValue>> dict = new Dictionary<TKey, List<TValue>>();
        /// <summary>
        /// 向一个键中添加值，支持为同一个键中添加多个值
        /// </summary>
        /// <param name="key"></param>
        /// <param name="value"></param>
        public void Add(TKey key, TValue value)
        {
            List<TValue> list;
            if (dict.TryGetValue(key, out list))
            {
                list.Add(value);
            }
            else
            {
                list = new List<TValue>();
                list.Add(value);
                dict.Add(key, list);
            }
        }
        /// <summary>
        /// 判断是否包含你要查找的键
        /// </summary>
        /// <param name="key"></param>
        /// <returns></returns>
        public bool ContainsKey(TKey key)
        {
            return dict.ContainsKey(key);
        }
        /// <summary>
        /// 返回字典中键的集合
        /// </summary>
        public ICollection<TKey> Keys
        {
            get
            {
                return dict.Keys;
            }
        }
        /// <summary>
        /// 此操作将移除键下面的所有值
        /// </summary>
        /// <param name="key"></param>
        /// <returns></returns>
        public bool Remove(TKey key)
        {
            return dict.Remove(key);
        }
        /// <summary>
        /// TryGetValue is not supported
        /// </summary>
        /// <param name="key"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        bool IDictionary<TKey, TValue>.TryGetValue(TKey key, out TValue value)
        {
            throw new NotSupportedException("TryGetValue is not supported");
        }
        /// <summary>
        /// 返回值的集合
        /// </summary>
        public ICollection<TValue> Values
        {
            get
            {
                List<TValue> values = new List<TValue>();
                Dictionary<TKey, List<TValue>>.Enumerator enumerator = dict.GetEnumerator();
                while (enumerator.MoveNext())
                {
                    values.AddRange(enumerator.Current.Value);
                }
                return values;
            }
        }
        /// <summary>
        /// TryGetValue is not supported
        /// </summary>
        /// <param name="key"></param>
        /// <returns></returns>
        TValue IDictionary<TKey, TValue>.this[TKey key]
        {
            get
            {
                if (dict.ContainsKey(key))
                {
                    return dict[key].FirstOrDefault();
                }
                else
                {
                    return default(TValue);
                }
                //throw new NotSupportedException(
                //"return single element by key is not supported");
            }
            set
            {
                List<TValue> list;
                if (dict.TryGetValue(key, out list))
                {
                    list.Add(value);
                }
                else
                {
                    list = new List<TValue>();
                    list.Add(value);
                    dict.Add(key, list);
                }
                //throw new NotSupportedException(
                //"return single element by key is not supported");
            }
        }
        /// <summary>
        /// 返回键对应的值的集合
        /// </summary>
        /// <param name="key"></param>
        /// <returns></returns>
        public IList<TValue> this[TKey key]
        {
            get
            {
                return dict[key];
            }
        }
        /// <summary>
        /// 向键对应的值集合中添加元素
        /// </summary>
        /// <param name="item"></param>
        public void Add(KeyValuePair<TKey, TValue> item)
        {
            Add(item.Key, item.Value);
        }
        /// <summary>
        /// 清空字典
        /// </summary>
        public void Clear()
        {
            dict.Clear();
        }
        /// <summary>
        /// 是否存在键值对
        /// </summary>
        /// <param name="item"></param>
        /// <returns></returns>
        public bool Contains(KeyValuePair<TKey, TValue> item)
        {
            List<TValue> list;
            if (!dict.TryGetValue(item.Key, out list))
            {
                return false;
            }
            else
            {
                return list.Contains(item.Value);
            }
        }
        /// <summary>
        /// 将字典覆盖到数组的指定位置后面
        /// </summary>
        /// <param name="array"></param>
        /// <param name="arrayIndex"></param>
        public void CopyTo(KeyValuePair<TKey, TValue>[] array, int arrayIndex)
        {
            if (array == null)
                throw new ArgumentNullException("array");
            if (arrayIndex < 0 || arrayIndex > array.Length)
                throw new ArgumentOutOfRangeException("array index out of range");
            if (array.Length - arrayIndex < this.Count)
                throw new ArgumentException("Array too small");
            Dictionary<TKey, List<TValue>>.Enumerator enumerator = dict.GetEnumerator();
            while (enumerator.MoveNext())
            {
                KeyValuePair<TKey, List<TValue>> mapPair = enumerator.Current;
                foreach (TValue val in mapPair.Value)
                {
                    array[arrayIndex++] = new KeyValuePair<TKey, TValue>(mapPair.Key, val);
                }
            }
        }
        /// <summary>
        /// 返回键值对总数
        /// </summary>
        public int Count
        {
            get
            {
                int count = 0;
                Dictionary<TKey, List<TValue>>.Enumerator enumerator = dict.GetEnumerator();
                while (enumerator.MoveNext())
                {
                    count += enumerator.Current.Value.Count;
                }
                return count;
            }
        }

        public bool IsReadOnly
        {
            get { return false; }
        }

        public bool Remove(KeyValuePair<TKey, TValue> item)
        {
            List<TValue> list;
            if (dict.TryGetValue(item.Key, out list))
            {
                return list.Remove(item.Value);
            }
            else
            {
                return false;
            }
        }
        /// <summary>
        /// 返回所有键值对的集合
        /// </summary>
        /// <returns></returns>
        public IEnumerator<KeyValuePair<TKey, TValue>> GetEnumerator()
        {
            Dictionary<TKey, List<TValue>>.Enumerator enumeratorKeys = dict.GetEnumerator();
            while (enumeratorKeys.MoveNext())
            {
                foreach (TValue item in enumeratorKeys.Current.Value)
                {
                    KeyValuePair<TKey, TValue> pair = new KeyValuePair<TKey, TValue>(enumeratorKeys.Current.Key, item);
                    yield return pair;
                }
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}
