using System.Collections;
using System.ComponentModel;
using System.Data;
using System.Reflection;

namespace ElincloudServer.Common
{
    public class DataTableUtil
    {
        /// <summary>
        /// 集合转Table
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="collection"></param>
        /// <returns></returns>
        public DataTable ToDataTable<T>(IEnumerable<T> collection, bool isAddColumnByDescripAttribute = false)
        {
            try
            {
                var props = typeof(T).GetProperties();
                var dt = new DataTable();
                if (isAddColumnByDescripAttribute)
                {
                    List<PropertyInfo> _tempProp = new List<PropertyInfo>();
                    foreach (var porp in props)
                    {
                        var descriptionAttributes = porp.GetCustomAttributes(typeof(DescriptionAttribute), false);
                        if (descriptionAttributes != null && descriptionAttributes.Length > 0)
                        {
                            var da = descriptionAttributes[0] as DescriptionAttribute;
                            if (da != null)
                            {
                                _tempProp.Add(porp);
                                dt.Columns.Add(new DataColumn(da.Description, porp.PropertyType));
                            }
                        }
                    }

                    if (collection.Count() > 0)
                    {
                        for (int i = 0; i < collection.Count(); i++)
                        {
                            ArrayList tempList = new ArrayList();
                            foreach (PropertyInfo pi in _tempProp)
                            {
                                object obj = pi.GetValue(collection.ElementAt(i), null);
                                tempList.Add(obj);
                            }
                            object[] array = tempList.ToArray();
                            dt.LoadDataRow(array, true);
                        }
                    }
                }
                else
                {
                    dt.Columns.AddRange(props.Select(p => new DataColumn(p.Name, p.PropertyType)).ToArray());
                    if (collection.Count() > 0)
                    {
                        for (int i = 0; i < collection.Count(); i++)
                        {
                            ArrayList tempList = new ArrayList();
                            foreach (PropertyInfo pi in props)
                            {
                                object obj = pi.GetValue(collection.ElementAt(i), null);
                                tempList.Add(obj);
                            }
                            object[] array = tempList.ToArray();
                            dt.LoadDataRow(array, true);
                        }
                    }
                }

                return dt;
            }
            catch (Exception ex)
            {
                throw ex;
            }
        }

        /// <summary>
        /// Table转集合
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="table"></param>
        /// <returns></returns>
        public IList<T> ConvertTo<T>(DataTable table)
        {
            if (table == null)
            {
                return null;
            }

            List<DataRow> rows = new List<DataRow>();

            foreach (DataRow row in table.Rows)
            {
                rows.Add(row);
            }

            return ConvertTo<T>(rows);
        }

        private IList<T> ConvertTo<T>(IList<DataRow> rows)
        {
            IList<T> list = null;

            if (rows != null)
            {
                list = new List<T>();

                foreach (DataRow row in rows)
                {
                    T item = CreateItem<T>(row);
                    list.Add(item);
                }
            }

            return list;
        }

        private T CreateItem<T>(DataRow row)
        {
            T obj = default(T);
            if (row != null)
            {
                obj = Activator.CreateInstance<T>();

                foreach (DataColumn column in row.Table.Columns)
                {
                    PropertyInfo prop = obj.GetType().GetProperty(column.ColumnName);
                    try
                    {
                        object value = row[column.ColumnName];
                        prop.SetValue(obj, value, null);
                    }
                    catch
                    {

                    }
                }
            }

            return obj;
        }
    }
}
