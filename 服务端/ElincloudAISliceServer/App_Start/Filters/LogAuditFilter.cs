using Microsoft.AspNetCore.Mvc.Controllers;
using Microsoft.AspNetCore.Mvc.Filters;
using Microsoft.AspNetCore.Mvc;
using Microsoft.IdentityModel.Tokens;
using Newtonsoft.Json;
using System.Diagnostics;
using ElincloudServer.Common;
using System.Reflection;
using ElincloudServer.Entity;
using ElincloudServer.BLServices;

namespace ElincloudServer
{
    public class LogAuditFilter : Attribute, IAsyncActionFilter
    {
        public async Task OnActionExecutionAsync(ActionExecutingContext context, ActionExecutionDelegate next)
        {
            if (!ShouldSaveAudit(context))
            {
                await next();
                return;
            }

            var type = (context.ActionDescriptor as ControllerActionDescriptor).ControllerTypeInfo.AsType();
            var method = (context.ActionDescriptor as ControllerActionDescriptor).MethodInfo;
            var arguments = context.ActionArguments;
            var stopwatch = Stopwatch.StartNew();

            var auditInfo = new AuditLogInfo
            {
                LoginName = IMCenterUtil.IMCenter.UserInfo.LoginName,
                UserName = IMCenterUtil.IMCenter.UserInfo.UserName,
                ServiceName = type != null ? type.FullName : "",
                //5.0版本以上，varchar(50)，指的是50字符，无论存放的是数字、字母还是UTF8汉字（每个汉字3字节），都可以存放50个。其他数据库要注意下这里
                MethodName = method.Name,
                Parameters = ConvertArgumentsToJson(arguments),
                ExecutionTime = DateTime.Now,
                BrowserInfo = Service.GetHeaderValue("User-Agent"),
                ClientIpAddress = Service.HttpContextAccessor.HttpContext.Connection.RemoteIpAddress.ToString(),
                ClientName = "",
            };

            ActionExecutedContext result = null;
            try
            {
                result = await next();
                if (result.Exception != null && !result.ExceptionHandled)
                {
                    auditInfo.Exception = result.Exception.ToString();
                    auditInfo.ExceptionMessage = result.Exception.Message;
                }
            }
            catch (Exception ex)
            {
                auditInfo.Exception = ex.ToString();
                auditInfo.ExceptionMessage = ex.Message;
            }
            finally
            {
                stopwatch.Stop();
                auditInfo.ExecutionDuration = Convert.ToInt32(stopwatch.Elapsed.TotalMilliseconds);

                if (result != null)
                {
                    //调用的方法为Get方法时不记录审计日志
                    if (string.IsNullOrWhiteSpace(auditInfo.MethodName)|| !auditInfo.MethodName.StartsWith("GET", StringComparison.OrdinalIgnoreCase))
                    {
                        switch (result.Result)
                        {
                            case ObjectResult objectResult:
                                auditInfo.ReturnValue = JsonConvert.SerializeObject(objectResult.Value);
                                break;

                            case JsonResult jsonResult:
                                auditInfo.ReturnValue = JsonConvert.SerializeObject(jsonResult.Value);
                                break;

                            case ContentResult contentResult:
                                auditInfo.ReturnValue = contentResult.Content;
                                break;
                        }
                    }
                }
                BLService.LogBLService.AddAuditLogInfo(auditInfo);
            }
        }

        private static bool ShouldSaveAudit(ActionExecutingContext context, bool defaultValue = false)
        {
            if (!(context.ActionDescriptor is ControllerActionDescriptor))
            {
                return false;
            }
            else
            {
                MethodInfo methodInfo = (context.ActionDescriptor as ControllerActionDescriptor).MethodInfo;
                if (methodInfo != null)
                {
                    //私有方法不用审计
                    if (!methodInfo.IsPublic)
                    {
                        return false;
                    }
                    //Action标记为审计的方法需要做日志审计
                    if (methodInfo.HasAttribute<AuditAttribute>())
                    {
                        return true;
                    }
                    //Action标记为不审计的方法无需做日志审计
                    if (methodInfo.HasAttribute<DisableAuditAttribute>())
                    {
                        return false;
                    }
                    var classType = methodInfo.DeclaringType;
                    if (classType != null)
                    {
                        //Controller标记为审计的方法需要做日志审计
                        if (classType.GetTypeInfo().HasAttribute<AuditAttribute>())
                        {
                            return true;
                        }
                        //Controller标记为不审计的方法无需做日志审计
                        if (classType.GetTypeInfo().HasAttribute<DisableAuditAttribute>())
                        {
                            return false;
                        }
                    }

                }
                else
                {
                    return false;
                }
            }
            return defaultValue;
        }

        private string ConvertArgumentsToJson(IDictionary<string, object> arguments)
        {
            try
            {
                if (arguments.IsNullOrEmpty())
                {
                    return "{}";
                }

                var dictionary = new Dictionary<string, object>();

                foreach (var argument in arguments)
                {
                    dictionary[argument.Key] = argument.Value;
                }

                return JsonConvert.SerializeObject(dictionary);
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog(ex.ToString(), ex);
                return "{}";
            }
        }
    }
}
