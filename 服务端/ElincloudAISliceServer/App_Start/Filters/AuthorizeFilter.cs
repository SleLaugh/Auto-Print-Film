using ElincloudServer.BLServices;
using ElincloudServer.Common;
using ElincloudServer.Entity;
using Microsoft.AspNetCore.Mvc.Filters;
using System.Net;


namespace ElincloudServer
{
    public class AuthorizeFilter : Attribute, IAuthorizationFilter
    {
        public void OnAuthorization(AuthorizationFilterContext context)
        {
            var env = context.HttpContext.RequestServices.GetService<IWebHostEnvironment>();
            bool isDevelopment = env?.IsDevelopment() == true;

            if (context.ActionDescriptor.EndpointMetadata.Any(m => m is SkipAuthorizeFilterAttribute))
            {
                return;
            }
            AuthorizeStatus status;
            if (context.HttpContext.Request.Method == "OPTIONS")//自定义Header时，引起跨域问题，导致请求2次  一次预请求，一次真正请求，预请求时获取不到自定义Header值，所以增加此代码
            {
                status = AuthorizeStatus.Success;
                //return true;
            }
            else
            {
                status = checkCustomTraceNo(context.HttpContext.Request.Headers[ApplicationEx.CustomTraceNo], context.HttpContext.Request.Headers[ApplicationEx.ActionAddress]);//高频请求验证
                if (status == AuthorizeStatus.Success)
                {
                    if (status == AuthorizeStatus.Success)//key验证成功再进行业务验证
                    {
                        string loginName = context.HttpContext.Request.Headers[ApplicationEx.LoginName];
                        loginName = loginName == null ? "" : loginName;
                        if (loginName != string.Empty)//需要登录的接口业务验证，不需要登录的接口无需再做业务验证
                        {
                            if (IMCenterUtil.IMCenter.DicClients.ContainsKey(loginName))
                            {
                                string token = context.HttpContext.Request.Headers[ApplicationEx.Token];
                                if (!string.IsNullOrEmpty(token))
                                {
                                    if (IMCenterUtil.IMCenter.DicClients[loginName].Token == token)
                                    {
                                        var cookieOptions = new CookieOptions()
                                        {
                                            Secure = false,
                                            SameSite = SameSiteMode.Lax,
                                            Expires = DateTime.Now.AddMinutes(30),
                                        };

                                        if (!isDevelopment)
                                        {
                                            string host = context.HttpContext.Request.Host.Host; // 如 "api.example.com" 或 "192.168.1.100"

                                            // 判断 host 是否为 IP 地址
                                            if (!IPAddress.TryParse(host, out _))
                                            {
                                                // 是域名，可以安全设置 Domain
                                                cookieOptions.Domain = Common.Utils.ConfigUtil.CookieDomain; // 如 ".example.com"
                                            }
                                            // 如果是 IP，则不设置 Domain（让 Cookie 仅作用于该 IP）
                                        }

                                        context.HttpContext.Response.Cookies.Append(ApplicationEx.Token, token, cookieOptions);
                                        status = AuthorizeStatus.Success;
                                        //return true;
                                    }
                                    else
                                    {
                                        //被别人挤掉了
                                        status = AuthorizeStatus.LoginOut;
                                    }
                                }
                                else
                                {
                                    status = AuthorizeStatus.TokenInvalid;
                                }
                            }
                            else
                            {
                                status = AuthorizeStatus.UnLogin;
                            }
                        }
                    }
                }
            }
            if (status != AuthorizeStatus.Success)//当status为success这里不返回，让请求直接到一下步。
            {
                context.Result = new Microsoft.AspNetCore.Mvc.JsonResult(new ClientResult()
                {
                    Success = status == AuthorizeStatus.Success,
                    Message = ElincloudServer.Common.Utils.EnumUtil.GetEnumDescriptionByName<AuthorizeStatus>(status),
                    Result = ((int)status).ToString()
                });
            }
        }

        private AuthorizeStatus checkCustomTraceNo(string customTraceNo, string actionAddress)
        {
            if (ElincloudServer.Common.Utils.StringUtil.MD5Encrypt(actionAddress + ApplicationEx.EncryptKey) == customTraceNo)
            {
                //验证逻辑：同一个ID在1s内请求次数超过5次则拒绝本次请求，这种请求请可能是自动化程序请求
                if (Service.UrlRequests.ContainsKey(customTraceNo))
                {
                    UrlRequestCache cache = Service.UrlRequests[customTraceNo];
                    if (cache != null)
                    {
                        if (DateTime.Now > cache.LastRequestDate.AddSeconds(1))//距上一次请求时间间隔超过1s，则直接更新缓存。
                        {
                            cache.LastRequestDate = DateTime.Now;
                            cache.RequestCount = 1;
                            return AuthorizeStatus.Success;
                        }
                        else
                        {
                            if (cache.RequestCount < 5)//距上一次请求时间间隔在1s内时，验证请求次数如果在5次以下则直接把请求次数+1。
                            {
                                cache.RequestCount += 1;
                                return AuthorizeStatus.Success;
                            }
                            else//距上一次请求时间间隔在1s内时，验证请求次数如果在大于5次则直接拒绝本次请求。
                            {
                                return AuthorizeStatus.RequestTooBusy;
                            }
                        }
                    }
                }
                else//如果缓存中没有信息，说明该ID的请求是第一次，直接缓存数据即可
                {
                    Service.UrlRequests.AddOrUpdate(customTraceNo, new UrlRequestCache()
                    {
                        RequestID = customTraceNo,
                        RequestCount = 1,
                        LastRequestDate = DateTime.Now,
                    }, (customTraceNo, urlRequestCache) =>
                    {
                        urlRequestCache = new UrlRequestCache()
                        {
                            RequestID = customTraceNo,
                            RequestCount = 1,
                            LastRequestDate = DateTime.Now,
                        };
                        return urlRequestCache;
                    });
                }

                return AuthorizeStatus.Success;
            }
            else
            {
                //当传过来的请求ID格式为空的时候，直接不允许请求，但这种情况可能不会出现
                return AuthorizeStatus.TimeOutOfSync;
            }
        }
    }
}
