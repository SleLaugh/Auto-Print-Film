using System.Text.Json;
using System.Text;
using Newtonsoft.Json;

namespace ElincloudServer.Common.CommonUtil
{
    /// <summary>
    /// 易联接口调用http工具类
    /// </summary>
    public static class ElincloudServerHttpHelperUtil
    {
        private static readonly HttpClient _client = new HttpClient
        {
            Timeout = TimeSpan.FromSeconds(30)
        };

        /// <summary>
        /// POST请求（同步，传对象）
        /// </summary>
        public static T PostJson<T>(string url, object data = null)
        {
            var settings = new Newtonsoft.Json.JsonSerializerSettings
            {
                DateFormatString = "yyyy-MM-dd HH:mm:ss"
            };

            string jsonData = data != null ? Newtonsoft.Json.JsonConvert.SerializeObject(data, settings) : "";
            if (!string.IsNullOrEmpty(jsonData))
            {
                using JsonDocument document = JsonDocument.Parse(jsonData);
                JsonElement root = document.RootElement;
                var encryptedData = new Dictionary<string, object>();
                foreach (JsonProperty property in root.EnumerateObject())
                {
                    encryptedData[property.Name] = Common.Utils.StringUtil.SM4EncryptECB(property.Value.ToString(), Common.Utils.StringUtil.Decrypt(Common.ApplicationEx.ParmEncryptKey, Common.Utils.StringUtil.Decrypt(Common.ApplicationEx.EncryptKey)));
                }
                jsonData = System.Text.Json.JsonSerializer.Serialize(encryptedData);
            }
            using HttpRequestMessage request = new HttpRequestMessage(HttpMethod.Post, url)
            {
                Content = new StringContent(jsonData, Encoding.UTF8, "application/json")
            };
            // 添加自定义头
            AddCustomHeaders(request, url, data);
            // 发送请求并获取响应
            HttpResponseMessage response = _client.Send(request);
            response.EnsureSuccessStatusCode();
            // 读取响应内容
            string jsonString = response.Content.ReadAsStringAsync().Result;

            if (!IsJson(jsonString))
            {
                // 去掉外层引号
                jsonString = jsonString.Trim('"');
                jsonString = Utils.StringUtil.Decrypt(jsonString);
            }
            else
            {
                // 如果响应中有转义字符，解码它们
                //  jsonString = System.Text.RegularExpressions.Regex.Unescape(jsonString);
            }
            return Newtonsoft.Json.JsonConvert.DeserializeObject<T>(jsonString);
        }
        public static bool IsJson(string input)
        {
            if (string.IsNullOrWhiteSpace(input)) return false;

            input = input.Trim();
            if (!(input.StartsWith("{") && input.EndsWith("}")) &&  // 对象
                !(input.StartsWith("[") && input.EndsWith("]")))    // 数组
                return false;

            try
            {
                Newtonsoft.Json.Linq.JToken.Parse(input);
                return true;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// 添加自定义头
        /// </summary>
        private static void AddCustomHeaders(HttpRequestMessage request, string url, object parms=null)
        {
            var settings = new Newtonsoft.Json.JsonSerializerSettings
            {
                DateFormatString = "yyyy-MM-dd HH:mm:ss"
            };

            string body = parms != null ? Newtonsoft.Json.JsonConvert.SerializeObject(parms, settings) : "";

            long timeStamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            int nonce = new Random().Next(100000, 200000);

            // 按字母顺序排序
            StringBuilder query = new StringBuilder();
            if (body != null)
            {
                IDictionary<string, string> sortedParams = new SortedDictionary<string, string>();
                Dictionary<string, object> keyValues = Newtonsoft.Json.JsonConvert.DeserializeObject<Dictionary<string, object>>(body);
                if (keyValues != null && keyValues.Count > 0)
                {
                    foreach (var item in keyValues)
                    {
                        sortedParams.Add(item.Key, Utils.StringUtil.IsJsonString(item.Value.ToString()) ? Newtonsoft.Json.JsonConvert.SerializeObject(item.Value, Formatting.None) : item.Value.ToString());
                    }
                    foreach (var sort in sortedParams)
                    {
                        query.Append(sort.Key.Trim());
                        string value = ElincloudServer.Common.Utils.StringUtil.InterceptString(sort.Value, 100);
                        query.Append(value);
                    }
                }

            }

            string data = query.ToString();
            // 生成签名
            string signString = $"{nonce}{timeStamp}{data}{Utils.StringUtil.Decrypt(ElincloudServer.Common.ApplicationEx.EncryptKey)}";
            string signature = Utils.StringUtil.MD5Encrypt(signString).ToUpper();
            string traceNo = Utils.StringUtil.MD5Encrypt(url + ElincloudServer.Common.ApplicationEx.EncryptKey).ToUpper();
            // 添加自定义头
            request.Headers.TryAddWithoutValidation(ApplicationEx.CustomTraceNo, traceNo);
            request.Headers.TryAddWithoutValidation(ApplicationEx.ActionAddress, url);
            request.Headers.TryAddWithoutValidation(ApplicationEx.TimeStamp, timeStamp.ToString());
            request.Headers.TryAddWithoutValidation(ApplicationEx.Nonce, nonce.ToString());
            request.Headers.TryAddWithoutValidation(ApplicationEx.Signature, signature);
            request.Headers.TryAddWithoutValidation(ApplicationEx.SecretKey, Common.ApplicationEx.ParmEncryptKey);
        }
    }
}
