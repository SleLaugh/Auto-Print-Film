using System.ComponentModel;
using System.Reflection;

namespace ElincloudServer.Common
{
    public class EnumUtil
    {

      
        public void SetEnumValue<TEntity, TEnum>(TEntity entity, int enumValue)
        {
            try
            {
                List<PropertyInfo> props = typeof(TEntity).GetProperties().ToList();
                foreach (var p in props)
                {
                    if (p.PropertyType == typeof(TEnum))
                    {
                        p.SetValue(entity, enumValue, null);
                        //判断枚举是否负值正确
                        if (p.GetValue(entity, null).ToString() == enumValue.ToString())
                        {
                            object val = Enum.GetValues(typeof(TEnum)).GetValue(0);
                            p.SetValue(entity, val, null);
                        }
                        break;
                    }
                }
            }
            catch (Exception)
            {
                throw new Exception("设置类型为[" + typeof(TEntity) + "]的类型为[" + typeof(TEnum) + "]的枚举值异常！");
            }
        }

        public TEnum GetEnumValue<TEnum>(int enumValue)
        {
            try
            {
                var vals = Enum.GetValues(typeof(TEnum));
                foreach (var item in vals)
                {
                    if ((int)item == enumValue)
                    {
                        return (TEnum)item;
                    }
                }
                return default(TEnum);
            }
            catch (Exception)
            {
                throw new Exception("获取类型[" + typeof(TEnum) + "]的枚举值异常！");
            }
        }

        public TEnum GetEnumValue<TEnum>(string enumValue)
        {
            try
            {
                var vals = Enum.GetValues(typeof(TEnum));
                foreach (var item in vals)
                {
                    if (item.ToString() == enumValue)
                    {
                        return (TEnum)Enum.Parse(typeof(TEnum), enumValue);
                    }
                }
                return default(TEnum);
            }
            catch (Exception)
            {
                throw new Exception("获取类型[" + typeof(TEnum) + "]的枚举值异常！");
            }
        }

        public string GetEnumDescriptionByName<TEnum>(TEnum enumItem)
        {
            FieldInfo fi = enumItem.GetType().GetField(enumItem.ToString());
            DescriptionAttribute[] attributes = (DescriptionAttribute[])fi.GetCustomAttributes(typeof(DescriptionAttribute), false);
            if (attributes != null && attributes.Length > 0)
            {
                return attributes[0].Description;
            }
            else
            {
                return enumItem.ToString();
            }
        }

        public List<TEnum> GetEnumList<TEnum>()
        {
            try
            {
                List<TEnum> list = Activator.CreateInstance<List<TEnum>>();
                var vals = Enum.GetValues(typeof(TEnum));
                foreach (TEnum item in vals)
                {
                    list.Add(item);
                }
                return list;
            }
            catch (Exception)
            {
                throw new Exception("获取类型[" + typeof(TEnum) + "]的枚举值集合异常！");
            }
        }
    }
}
