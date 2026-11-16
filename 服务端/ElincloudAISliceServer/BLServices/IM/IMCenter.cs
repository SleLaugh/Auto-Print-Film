using ElincloudServer.Common;
using ElincloudServer.Entity;
using System.Net;

namespace ElincloudServer.BLServices
{
    public class IMCenter
    {
        public IMCenter()
        {
            //初始化服务调用权限验证方法
            BeforeExcute = new Func<PermissionInfo>(() =>
            {
                string loginName = Service.GetHeaderValue(ApplicationEx.LoginName);
                if (!string.IsNullOrEmpty(loginName))
                {
                    KeyValuePair<string, IMClientInfo> kv = DicClients.ToArray().FirstOrDefault(p => p.Key == loginName);
                    bool passed = kv.Key != null && kv.Value != null;
                    return new PermissionInfo()
                    {
                        IsPermitted = passed,
                        LoginName = passed ? kv.Value.LoginName : "",
                        UserName = passed ? kv.Value.UserName : "",
                        Password = passed ? kv.Value.Password : "",
                        UserID = passed ? kv.Value.UserID : "",
                        IsAdmin = passed ? kv.Value.IsAdmin : false,
                        HospitalID = passed ? kv.Value.HospitalID : "",
                        HospitalCode = passed ? kv.Value.HospitalCode : "",
                        HospitalName = passed ? kv.Value.HospitalName : "",
                        OrganizationID = passed ? kv.Value.OrganizationID : "",
                        OrganizationName = passed ? kv.Value.OrganizationName : "",
                        HospitalGroupID = passed ? kv.Value.HospitalGroupID : "",
                        HospitalGroupName = passed ? kv.Value.HospitalGroupName : "",
                        HospitalIDs=passed?kv.Value.HospitalIDs:null
                    };
                }
                else
                {
                    return new PermissionInfo() { IsPermitted = false };
                }
            });
        }

        Mutex locker = new Mutex(false);
        private Dictionary<string, IMClientInfo> dicClients = new Dictionary<string, IMClientInfo>();
        public Dictionary<string, IMClientInfo> DicClients
        {
            get
            {
                try
                {
                    locker.WaitOne();
                    return dicClients;
                }
                finally
                {
                    locker.ReleaseMutex();
                }
            }
        }

        public IMClientInfo UserInfo
        {
            get
            {
                string loginName = Service.GetHeaderValue(ApplicationEx.LoginName);
                if (!string.IsNullOrEmpty(loginName))
                {
                    if (DicClients.ContainsKey(loginName))
                    {
                        return DicClients[loginName];
                    }
                }
                return new IMClientInfo();
            }
        }

        public Action<string, IMClientInfo> SetDisplayMessageAction;
        public Action<LoginInfo> AddNodeAction;
        public Action<string> RemoveNodeAction;

        /// <summary>
        /// 在调用操作之前执行的委托，提供权限上下文。
        /// </summary>
        public Func<PermissionInfo> BeforeExcute;

        #region 用户上线事件

        public void RegistClient(ClientResult userRet)
        {
            AccountInfo accountInfo = userRet.Result == null ? null : userRet.Result as AccountInfo;
            try
            {
                if (userRet.Success)
                {
                    IIMServiceCallBack callBack = null;//TODO:OperationContext.Current.GetCallbackChannel<IIMServiceCallBack>();
                    string loginName = accountInfo.LoginInfo.LoginName;
                    //TODO: OperationContext.Current.Channel.Closed += Channel_Closed;//用户主动离线
                    if (DicClients.ContainsKey(loginName))//用户已经在别的地方登陆了 把他挤下去
                    {
                        SendMessage(loginName, "您的帐号已经在别的地方登陆,如果不是您自己操作请尽快修改密码或联系管理员！", "Service", loginName, SysMessages.ReLogin, "帐号安全");
                        DicClients.Remove(loginName);
                        SetDisplayMessage(string.Format("用户离线,登录帐户号 [{0}]", loginName));
                        RemoveNode(loginName);
                    }
                    string token = ElincloudServer.Common.Utils.StringUtil.NewGuid().ToString();
                    KeyValuePair<string, IMClientInfo> newUser = new KeyValuePair<string, IMClientInfo>(loginName, new IMClientInfo()
                    {
                        Token = token,
                        UserID = accountInfo.LoginInfo.UserID,
                        UserName = accountInfo.LoginInfo.UserName,
                        LoginName = accountInfo.LoginInfo.LoginName,
                        Password = accountInfo.LoginInfo.Password,
                        IsAdmin = accountInfo.LoginInfo.IsAdmin,
                        OrganizationID = accountInfo.UserOrganization != null ? accountInfo.UserOrganization.ID : "",
                        OrganizationName = accountInfo.UserOrganization != null ? accountInfo.UserOrganization.Name : "",
                        HospitalID = accountInfo.UserHospital != null ? accountInfo.UserHospital.ID : "",
                        HospitalCode = accountInfo.UserHospital != null ? accountInfo.UserHospital.Code : "",
                        HospitalName = accountInfo.UserHospital != null ? accountInfo.UserHospital.Name : "",
                        HospitalGroupID = accountInfo.UserHospital != null ? accountInfo.UserHospital.HospitalGroupID : "",
                        HospitalGroupName = accountInfo.UserHospital != null ? accountInfo.UserHospital.HospitalGroupName : "",
                        HospitalIDs=accountInfo.LoginInfo.HospitalIDs,
                        CallBack = callBack,
                        //TODO: Channel = OperationContext.Current.Channel,
                    });
                    DicClients.Add(newUser.Key, newUser.Value);
                    SetDisplayMessage(string.Format("用户上线,登录帐号 [{0}]]", accountInfo.LoginInfo.LoginName));
                    // 设置响应前端的cookie
                    var cookieOptions = new CookieOptions()
                    {
                        Secure = false,
                        SameSite = SameSiteMode.Lax,
                        Expires = DateTime.Now.AddMinutes(30),
                    };

                    var env = Service.HttpContextAccessor.HttpContext.RequestServices.GetService<IWebHostEnvironment>();
                    bool isDevelopment = env?.IsDevelopment() == true;

                    if (!isDevelopment)
                    {
                        string host = Service.HttpContextAccessor.HttpContext.Request.Host.Host; // 如 "api.example.com" 或 "192.168.1.100"

                        // 判断 host 是否为 IP 地址
                        if (!IPAddress.TryParse(host, out _))
                        {
                            // 是域名，可以安全设置 Domain
                            cookieOptions.Domain = Common.Utils.ConfigUtil.CookieDomain; // 如 ".example.com"
                        }
                        // 如果是 IP，则不设置 Domain（让 Cookie 仅作用于该 IP）
                    }

                    Service.HttpContextAccessor.HttpContext.Response.Cookies.Append(ApplicationEx.Token, token, cookieOptions);
                    AddNode(newUser);
                }
                else
                {
                    if (accountInfo != null)
                    {
                        SetDisplayMessage(string.Format("帐户[{0}]登录失败！失败原因:{1}", accountInfo.LoginInfo.LoginName, userRet.Message));
                    }
                }
            }
            catch (Exception ex)
            {
                SetDisplayMessage("服务异常！[" + ex.Message + "]", new IMClientInfo()
                {
                    UserID = accountInfo == null ? "" : accountInfo.LoginInfo.UserID,
                    UserName = accountInfo == null ? "" : accountInfo.LoginInfo.UserName,
                    LoginName = accountInfo == null ? "" : accountInfo.LoginInfo.LoginName,
                    Password = accountInfo == null ? "" : accountInfo.LoginInfo.Password,
                    IsAdmin = accountInfo == null ? false : accountInfo.LoginInfo.IsAdmin,
                    //OrganizationID = accountInfo == null ? "" : accountInfo.UserOrganization.ID,
                    //OrganizationName = accountInfo == null ? "" : accountInfo.UserOrganization.Name,
                    //HospitalID = accountInfo == null ? "" : accountInfo.UserHospital.ID,
                    //HospitalName = accountInfo == null ? "" : accountInfo.UserHospital.Name,
                });
            }

        }

        #endregion

        #region 用户离线事件

        public void UpdateLogoutTokens(PermissionInfo pi)
        {
            if (DicClients.ContainsKey(pi.LoginName))
            {
                try
                {
                    KeyValuePair<string, IMClientInfo> kv = DicClients.ToArray().FirstOrDefault(p => p.Value.LoginName == pi.LoginName);
                    if (kv.Key != null)
                    {
                        DicClients.Remove(kv.Key);
                        SetDisplayMessage(string.Format("用户离线,登录帐户号 [{0}]", kv.Value.LoginName));
                        RemoveNode(kv.Key);
                    }
                    // 设置响应前端的cookie
                    if (Service.HttpContextAccessor.HttpContext.Response != null)
                    {
                        Service.HttpContextAccessor.HttpContext.Response.Cookies.Delete(ApplicationEx.Token);
                    }
                }
                catch (Exception ex)
                {
                    SetDisplayMessage(string.Format("注销用户失败![{0}]", ex.Message));
                }
            }
        }
        #endregion

        #region 向服务界面推送消息

        private void SetDisplayMessage(string msg, IMClientInfo clientInfo = null)
        {
            if (SetDisplayMessageAction != null)
            {
                IMClientInfo info = clientInfo;
                if (Service.HttpContextAccessor.HttpContext != null)
                {
                    string sessionid = Service.HttpContextAccessor.HttpContext.Session.Id;
                    if (sessionid != null)
                    {
                        if (DicClients.ContainsKey(sessionid))
                        {
                            info = DicClients[sessionid];
                        }
                    }
                }
                SetDisplayMessageAction(msg, info);
            }
        }

        private void AddNode(KeyValuePair<string, IMClientInfo> newUser)
        {
            if (AddNodeAction != null)
            {
                AddNodeAction(newUser.Value);
            }
        }

        private void RemoveNode(string key)
        {
            if (RemoveNodeAction != null)
            {
                RemoveNodeAction(key);
            }
        }

        public void ClearClient()
        {
            if (DicClients.Count > 0)
            {
                foreach (string client in DicClients.Keys.ToArray())
                {
                    SendMessage(client, string.Format("{0}已停止运行,请及时保存,以免数据丢失！", "服务"), "Service");
                    RemoveNodeAction(client);
                }
                DicClients.Clear();
            }
        }
        #endregion

        #region 向客户端推送消息

        public void BroadCast(string message)
        {
            try
            {
                if (message != string.Empty)
                {
                    foreach (string client in DicClients.Keys.ToArray())
                    {
                        SendMessage(client, message, "Service");
                    }
                }
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("广播消息异常！", ex);
            }

        }

        public void SendMessage(string targetUserLoginName, SysMessageInfo message)
        {
            SendMessage(targetUserLoginName, message.Content, message.Sender, message.ID, message.MessageType, message.Title);
        }

        public void SendMessage(string targetUserLoginName, string msg, string sender, string msgID = "0", SysMessages msgType = SysMessages.News, string title = "服务器消息")
        {
            KeyValuePair<string, IMClientInfo> kv = DicClients.ToArray().FirstOrDefault(p => p.Value.LoginName == targetUserLoginName);
            if (kv.Value != null)
            {
                try
                {
                    SysMessageInfo info = new SysMessageInfo()
                    {
                        Sender = sender,
                        MessageType = msgType,
                        ID = msgID,
                        Content = msg,
                        Title = title,
                        Date = DateTime.Now
                    };
                    if (kv.Value.CallBack != null)
                    {
                        kv.Value.CallBack.SendMessage(info);
                    }
                }
                catch (Exception ex)
                {
                    ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("为用户[" + targetUserLoginName + "]推送消息[" + msg + "]异常！", ex);
                    //TODO:RemoveChannel(kv.Value.CallBack);
                }
            }
        }

        #endregion

        #region 防服务被暴力破解算法

        private Dictionary<string, LoginSecurityInfo> userSecurityCache = new Dictionary<string, LoginSecurityInfo>();

        public bool LoginEnabled(string loginName)
        {
            if (userSecurityCache.ContainsKey(loginName))
            {
                LoginSecurityInfo info = userSecurityCache[loginName];
                if (info.IsInDeniedDuration)
                {
                    return false;
                }
            }
            return true;
        }

        public ClientResult SecurityChecked(string userLoginName, bool loginSuccess)
        {
            lock (userSecurityCache)
            {
                //验证规则:1分钟内连续登录失败5次或登录总数超过10次,则限制登录3分钟
                ClientResult ret = new ClientResult();
                if (!userSecurityCache.ContainsKey(userLoginName))//首次登录直接添加缓存
                {
                    LoginSecurityInfo info = new LoginSecurityInfo();
                    info.UserLoginName = userLoginName;
                    if (!info.IsInDeniedDuration)
                    {
                        info.LoginCount++;
                    }
                    if (!loginSuccess)
                    {
                        info.LoginFailedCount++;
                    }
                    else
                    {
                        info.LoginFailedCount = 0;
                    }
                    info.StartDate = DateTime.Now;
                    userSecurityCache.Add(userLoginName, info);
                    ret.Success = true;
                }
                else//多次登录,按规则验证
                {
                    LoginSecurityInfo info = userSecurityCache[userLoginName];
                    info.StartDate = DateTime.Now;

                    if (info.IsInValidateDuration)//计时周期在1分钟内
                    {

                        info.LoginCount++;

                        if (!loginSuccess)
                        {
                            info.LoginFailedCount++;
                        }
                        else
                        {
                            info.LoginFailedCount = 0;
                        }
                        if (info.LoginCount <= 10)//1分钟内累计登录不超过10次,验证是否有连续登录失败的情况
                        {

                            if (info.LoginFailedCount <= 5)
                            {
                                ret.Success = true;
                                if (info.LoginFailedCount == 5)
                                {
                                    info.LoginDeniedDate = DateTime.Now;
                                }
                            }
                            else//在10次登录限制内连续登录失败5次
                            {

                                //剩余限制时间
                                if (info.IsInDeniedDuration)//在限制时间内
                                {
                                    ret.Success = false;
                                    ret.Message = string.Format("短时间内登录失败次数过多,将于[{0}分{1}秒]之内限制登录!", info.LoginDeniedRemainingTime.Minutes, info.LoginDeniedRemainingTime.Seconds);
                                }
                                else//超过限制时间 用户可以重新登录
                                {
                                    info.LoginCount = 1;
                                    info.LoginFailedCount = 0;
                                    ret.Success = true;
                                }
                            }

                            if (info.LoginCount == 10)
                            {
                                info.LoginDeniedDate = DateTime.Now;
                            }
                        }

                        else//1分钟内累计登录达到10次,第11次登录的时候给前端返回消息在限定时间内不得登录
                        {

                            //剩余限制时间
                            if (info.IsInDeniedDuration)//在限制时间内
                            {
                                ret.Success = false;
                                ret.Message = string.Format("短时间内登录过于频繁,将于[{0}分{1}秒]之内限制登录!", info.LoginDeniedRemainingTime.Minutes, info.LoginDeniedRemainingTime.Seconds);
                            }
                            else//超过限制时间 用户可以重新登录
                            {
                                info.LoginCount = 1;
                                info.LoginFailedCount = 0;
                                ret.Success = true;
                            }
                        }
                    }
                    else//计时周期超过1分钟,重置验证状态
                    {
                        info.LoginCount = 1;
                        info.LoginFailedCount = 0;
                        ret.Success = true;
                    }
                }
                return ret;
            }
        }

        #endregion

    }
}
