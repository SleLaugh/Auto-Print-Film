using ElincloudServer.BLServices;
using ElincloudServer.Common;
using ElincloudServer.Entity;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Filters;
using Newtonsoft.Json;
using System.Security;
using System.Text;

namespace ElincloudServer
{
    public class ActionSecurityFilter : Attribute, IActionFilter
    {
        // do something after the action executes
        public void OnActionExecuted(ActionExecutedContext context)
        {
            var skipFilterAttribute = context.ActionDescriptor.EndpointMetadata.FirstOrDefault(m => m is SkipActionSecurityFilterAttribute);
            if (skipFilterAttribute == null)
            {
                //这里可以对返回前端的结果进行加密处理
                if (!ElincloudServer.Common.Utils.ConfigUtil.IsDebug)
                {
                    Microsoft.AspNetCore.Mvc.JsonResult jr = context.Result as Microsoft.AspNetCore.Mvc.JsonResult;
                    if (jr.Value is ClientResult)
                    {
                        ClientResult result = jr.Value as ClientResult;
                        if (result != null)
                        {
                            string json = JsonConvert.SerializeObject(result);
                            jr.Value = ElincloudServer.Common.Utils.StringUtil.Encrypt(json);
                        }
                    }
                }
            }
        }

        // do something before the action executes
        public void OnActionExecuting(ActionExecutingContext context)
        {
            try
            {
                var skipFilterAttribute = context.ActionDescriptor.EndpointMetadata.FirstOrDefault(m => m is SkipActionSecurityFilterAttribute);
                if (skipFilterAttribute == null)
                {
                    var request = context.HttpContext.Request;

                    var method = request.Method;
                    string timestamp = string.Empty;
                    string nonce = string.Empty;
                    string signature = string.Empty;

                    #region TimeStamp

                    if (request.Headers.ContainsKey(ApplicationEx.TimeStamp))
                    {
                        timestamp = request.Headers[ApplicationEx.TimeStamp].FirstOrDefault();
                        if (!string.IsNullOrEmpty(timestamp))
                        {
                            #region 验证请求的时间戳超过5mins则验证失败

                            double ts = 0;
                            bool timespanvalidate = double.TryParse(timestamp, out ts);
                            if (!timespanvalidate)
                            {
                                throw new SecurityException("无效的参数TimeStamp！");
                            }
                            else
                            {
                                double timestampDiff = (DateTime.UtcNow - new DateTime(1970, 1, 1, 0, 0, 0, 0)).TotalMilliseconds - ts;
                                bool falg = timestampDiff > 300 * 1000;
                                if (falg)
                                {
                                    throw new SecurityException("请求链接失效！(服务器与客户端时间差异过大)");
                                }
                            }

                            #endregion
                        }
                        else
                        {
                            throw new SecurityException("参数TimeStamp不能为空！");
                        }
                    }
                    else
                    {
                        throw new SecurityException("请求链接缺少参数TimeStamp！");
                    }

                    #endregion

                    #region Nonce

                    if (request.Headers.ContainsKey(ApplicationEx.Nonce))
                    {
                        nonce = request.Headers[ApplicationEx.Nonce].FirstOrDefault();
                        if (string.IsNullOrEmpty(nonce))
                        {
                            throw new SecurityException("参数nonce不能为空!");
                        }
                    }
                    else
                    {
                        throw new SecurityException("请求链接缺少参数nonce!");
                    }

                    #endregion

                    #region Signature

                    if (request.Headers.ContainsKey(ApplicationEx.Signature))
                    {
                        signature = request.Headers[ApplicationEx.Signature].FirstOrDefault();
                        if (string.IsNullOrEmpty(signature))
                        {
                            throw new SecurityException("参数Signature不能为空！");
                        }
                    }
                    else
                    {
                        throw new SecurityException("请求链接缺少参数Signature");
                    }

                    #endregion

                    #region key

                    string SecretKey=string.Empty;
                    if (request.Headers.ContainsKey(ApplicationEx.SecretKey))
                    {
                        SecretKey = request.Headers[ApplicationEx.SecretKey].FirstOrDefault();
                    }

                    #endregion

                    #region data

                    IDictionary<string, string> sortedParams = getBodyParaDic(request, SecretKey);

                    StringBuilder query = new StringBuilder();

                    foreach (var sort in sortedParams)
                    {
                        //query.Append(sort.Key.Trim()).Append(sort.Value.Trim());
                        query.Append(sort.Key.Trim());
                        string value = ElincloudServer.Common.Utils.StringUtil.InterceptString(sort.Value, 100);
                        query.Append(value);
                    }
                    string data = query.ToString();

                    #endregion

                    #region 验证请求

                    string key = ElincloudServer.Common.Utils.StringUtil.Decrypt(ElincloudServer.Common.ApplicationEx.EncryptKey);
                    var md5Staff = ElincloudServer.Common.Utils.StringUtil.MD5Encrypt(string.Concat(nonce + timestamp + data + key));

                    if (!md5Staff.Equals(signature))
                    {
                        throw new SecurityException("无效的签名！");
                    }

                    //下面的代码主要解决用body传参时，ActionArguments里面没有内容，这里主动填充后，Controller里面才能接收到对应的参数
                    foreach (var item in sortedParams)
                    {
                        if (!context.ActionArguments.ContainsKey(item.Key))
                        {
                            context.ActionArguments.Add(item.Key, item.Value);
                        }
                        else
                        {
                            context.ActionArguments[item.Key] = ElincloudServer.Common.Utils.JsonUtil.FromJson(context.ActionArguments[item.Key].GetType(), item.Value);
                        }
                    }

                    #endregion

                }
            }
            catch (Exception ex)
            {
                context.Result = new JsonResult(new ClientResult()
                {
                    Success = false,
                    Message = ex.Message,
                });
            }
        }

        private IDictionary<string, string> getBodyParaDic(HttpRequest request,string secretKey)
        {
            IDictionary<string, string> sortedParams = new SortedDictionary<string, string>();
            request.EnableBuffering();
            var stream = request.HttpContext.Request.Body;
            if (request.HttpContext.Request.ContentLength != null && request.HttpContext.Request.ContentLength > 0)
            {
                string jsonStr = "";
                using (StreamReader streamReader = new StreamReader(stream, Encoding.UTF8))
                {
                    try
                    {
                        jsonStr = streamReader.ReadToEndAsync().Result;
                    }
                    finally
                    {
                        request.HttpContext.Request.Body.Position = 0;
                    }
                }
                if (!string.IsNullOrWhiteSpace(secretKey))
                {
                    secretKey = Common.Utils.StringUtil.Decrypt(secretKey, ElincloudServer.Common.Utils.StringUtil.Decrypt(ElincloudServer.Common.ApplicationEx.EncryptKey));
                }
                //由于上传文件时使用的contentType为form-data，jsonStr为空，所以此处得做空验证
                if (!string.IsNullOrEmpty(jsonStr))
                {
                    Dictionary<string, object> keyValues = JsonConvert.DeserializeObject<Dictionary<string, object>>(jsonStr);
                    foreach (var item in keyValues)
                    {
                        string value = item.Value.ToString();
                        //验证参数中是否包含Sql注入字符
                        if (!ElincloudServer.Common.Utils.StringUtil.CheckIsSqlInjection(value))
                        {
                            if (!string.IsNullOrWhiteSpace(secretKey))
                            {
                                value = Common.Utils.StringUtil.SM4DecryptECB(value,secretKey);
                            }
                            sortedParams.Add(item.Key,value);
                        }
                        else
                        {
                            throw new SecurityException("参数包含Sql注入字符！");
                        }
                    }
                }
            }
            return sortedParams;
        }
    }
}
