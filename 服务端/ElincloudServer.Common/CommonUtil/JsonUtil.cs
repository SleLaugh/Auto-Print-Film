using Newtonsoft.Json;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;
using System.Text;
using System.Text.RegularExpressions;

namespace ElincloudServer.Common
{
    public class JsonUtil
    {
        /// <summary>
        /// 带命名空间的json字符串转实体
        /// </summary>
        public T FromJson<T>(string json)
        {
            try
            {
                //json = JsonDateTimeFormat(json);
                JsonSerializerSettings jsonSerializerSettings = new JsonSerializerSettings();
                jsonSerializerSettings.TypeNameHandling = TypeNameHandling.All;
                return JsonConvert.DeserializeObject<T>(json, jsonSerializerSettings);
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("Json转对像异常！", ex);
                return default(T);
            }

        }

        public object FromJson(Type type, string json)
        {
            try
            {
                if (string.IsNullOrEmpty(json))
                {
                    return null;
                }
                DataContractJsonSerializerSettings settings = new DataContractJsonSerializerSettings
                {
                    DateTimeFormat = new DateTimeFormat("yyyy-MM-dd HH:mm:ss")
                };

                DataContractJsonSerializer serializer = new DataContractJsonSerializer(type, settings);
                MemoryStream ms = new MemoryStream(Encoding.UTF8.GetBytes(json));
                return serializer.ReadObject(ms);
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("Json转对像异常!", ex);
                return null;
            }
        }

        /// <summary>
        /// 不带命名空间的json字符串转实体
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="jsonStr"></param>
        /// <returns></returns>
        public T FromJsonWithoutNameSpace<T>(string jsonStr)
        {
            try
            {
                DataContractJsonSerializer serializer = new DataContractJsonSerializer(typeof(T));
                MemoryStream ms = new MemoryStream(Encoding.UTF8.GetBytes(jsonStr));
                object obj = serializer.ReadObject(ms);
                return (T)obj;
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("Json转对像异常!", ex);
                return default(T);
            }
        }

        /// <summary>
        /// 返回带命名空间的Json字符串格式
        /// </summary>
        /// <param name="object"></param>
        /// <returns></returns>
        public string ToJson(object @object)
        {
            try
            {
                //IsoDateTimeConverter timeConverter = new IsoDateTimeConverter();
                ////这里使用自定义日期格式，如果不使用的话，默认是ISO8601格式
                //timeConverter.DateTimeFormat = "yyyy-MM-dd HH:mm:ss";
                JsonSerializerSettings jsonSerializerSettings = new JsonSerializerSettings();
                jsonSerializerSettings.TypeNameHandling = TypeNameHandling.All;
                string json = JsonConvert.SerializeObject(@object, Formatting.Indented, jsonSerializerSettings);
                return JsonDateTimeFormat(json);
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("对像转Json异常！", ex);
                return string.Empty;
            }

        }

        /// <summary>
        /// 返回不带命名空间的json字符串
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="obj"></param>
        /// <returns></returns>
        public string ToJsonWithoutNameSpace<T>(object obj)
        {
            try
            {
                DataContractJsonSerializer json = new DataContractJsonSerializer(typeof(T));
                MemoryStream ms = new MemoryStream();
                json.WriteObject(ms, obj);
                byte[] dataBytes = new byte[ms.Length];
                ms.Position = 0;
                ms.Read(dataBytes, 0, (int)ms.Length);
                string jsonStr = Encoding.UTF8.GetString(dataBytes);
                return jsonStr;
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("对像转Json异常!", ex);
                return null;
            }

        }

        /// <summary>
        /// 处理Json的时间格式为正常格式
        /// </summary>
        private string JsonDateTimeFormat(string json)
        {
            json = Regex.Replace(json,
                @"\\/Date\((\d+)\)\\/",
                match =>
                {
                    DateTime dt = new DateTime(1970, 1, 1);
                    dt = dt.AddMilliseconds(long.Parse(match.Groups[1].Value));
                    dt = dt.ToLocalTime();
                    return dt.ToString("yyyy-MM-dd HH:mm:ss.fff");
                });
            return json;
        }


    }
}
