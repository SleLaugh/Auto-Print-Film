using ElincloudServer.Entity;

namespace ElincloudServer.Common
{
    public class SMSUtil
    {
        public static string CloudImageRegisterStr = "云影像登录验证码";
        public static string UserRegisterStr = "用户注册验证码";


        /// <summary>
        /// 短息模版list
        /// </summary>
        /// <remarks>
        /// sle 2025年2月19日13:59:17
        /// </remarks>
        public readonly List<KeyValueModel> SMSTemplateList = new List<KeyValueModel>()
        {
            new KeyValueModel(){Key = CloudImageRegisterStr,Value="验证码，验证码有效期（分钟），医院名称" },
            new KeyValueModel(){Key = UserRegisterStr,Value="验证码，验证码有效期（分钟），医院名称" },
            new KeyValueModel(){Key = "授权期限预警",Value="医院名称，授权期限（天）" },
            new KeyValueModel(){Key = "磁盘使用率预警",Value="云主机名称，磁盘使用率" },
            new KeyValueModel(){Key = "CPU使用率预警",Value="云主机名称，CPU使用率" },
            new KeyValueModel(){Key = "原始影像未上传预警",Value="医院名称，多久未上传" },
            new KeyValueModel(){Key = "胶片未上传预警",Value="医院名称，多久未上传" },
            new KeyValueModel(){Key = "报告未上传预警",Value="医院名称，多久未上传" },
            new KeyValueModel(){Key = "前置机CPU使用率预警",Value="医院名称，CPU使用率" },
            new KeyValueModel(){Key = "前置机磁盘使用率预警",Value="医院名称，磁盘使用率" },
            new KeyValueModel(){Key = "前置机图像积压预警",Value="医院名称，图像数量" },
            new KeyValueModel(){Key = "前置机报告同步延时预警",Value="医院名称，系统类型，慢多久（分钟）" },
            new KeyValueModel(){Key = "云影像手机登录地址",Value="云影像地址，自定义参数" },
            new KeyValueModel(){Key = "云影像安全码登录地址",Value="云影像地址，安全码，自定义参数" },
            new KeyValueModel(){Key = "归档预警",Value="云主机名称，积压上限" },
            new KeyValueModel(){Key = "前置机运行状态预警",Value="医院名称，几个服务，多久不运行" }
        };

        public string SendVerificationCodeTemplateID = "1719547";
        ////云影像前置机图像积压预警
        //public string SendDcmFileTooMuchTemplateID = "1998617";
        ////云影像前置机数据同步异常预警
        //public string SendDataSynchronizationWarningTemplateID = "2320873";
        ////云影像前置机磁盘异常预警
        //public string SendDiskWarningTemplateID = "2463665";
        ////云影像前置机CPU异常预警
        //public string SendCpuWarningTemplateID = "2463666";
        private SMSConfigInfo smsConfigInfo = new SMSConfigInfo();

        #region 云影像登录验证码

        private int verificationCodeAvailableTime = 1;//1 min
        private Dictionary<string, VerificationCodeInfo> verificationCodes = new Dictionary<string, VerificationCodeInfo>();

        public string CheckVerificationCode(string telePhone, string verificationCode)
        {
            try
            {
                if (verificationCodes.ContainsKey(telePhone))
                {
                    VerificationCodeInfo info = verificationCodes[telePhone];
                    if ((DateTime.Now - info.SendTime).TotalMinutes < verificationCodeAvailableTime)
                    {
                        if (info.Code != verificationCode)
                        {
                            return "验证码错误！";
                        }
                        else
                        {
                            //验证通过 移除验证码
                            verificationCodes.Remove(telePhone);
                        }
                    }
                    else
                    {
                        verificationCodes.Remove(telePhone);
                        return "验证码已过期,请重新获取！";
                    }
                }
                else
                {
                    return "请先获取验证码！";
                }
                return string.Empty;
            }
            catch (Exception ex)
            {
                string ret = "校验手机号和验证码异常！";
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog(ret, ex);
                return ret;
            }
        }

        public Entity.ClientResult SendVerificationCodeToTelePhone(string telePhone, string hospitalName,string key)
        {
            Entity.ClientResult ret = new Entity.ClientResult() { Success = true };
            try
            {
                if (verificationCodes.ContainsKey(telePhone))
                {
                    VerificationCodeInfo info = verificationCodes[telePhone];
                    if ((DateTime.Now - info.SendTime).TotalMinutes < verificationCodeAvailableTime)
                    {
                        return new Entity.ClientResult() { Success = true };
                    }
                    else
                    {
                        verificationCodes.Remove(telePhone);
                    }
                }
                string vc = ElincloudServer.Common.Utils.StringUtil.GetRandomCode(6);
                List<string> param = new List<string>() { vc, verificationCodeAvailableTime.ToString(), hospitalName };
                if (SMS != null)
                {
                    KeyValueModel info = smsConfigInfo.SMSTemplateList.FirstOrDefault(p => p.Key == key);
                    if (info != null)
                    {
                        Dictionary<string, object> retData = SMS.SendTemplateSMS(telePhone, info.Value, param.ToArray());
                        if (retData["statusMsg"].ToString() != "成功")
                        {
                            string msg = retData["statusMsg"].ToString();
                            ElincloudServer.Common.Utils.LogUtil.RecordErrorLog(msg);
                            return new Entity.ClientResult() { Success = false, Message = msg };
                        }
                        else
                        {
                            verificationCodes.Add(telePhone, new VerificationCodeInfo()
                            {
                                TelePhone = telePhone,
                                Code = vc,
                                SendTime = DateTime.Now
                            });
                            return new Entity.ClientResult() { Success = true };
                        }
                    }
                    else
                    {
                        ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("未找到[云影像登录验证码]的短信模板信息!");
                        return new Entity.ClientResult() { Success = false, Message = "发送短信失败，未找到对应的模板信息!" };
                    }
                }
                else
                {
                    return new Entity.ClientResult() { Success = false, Message = "初始化短信服务失败!" };
                }
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("发短信异常！", ex);
                return new Entity.ClientResult() { Success = false, Message = ex.Message };
            }
        }

        #endregion

        #region 发送短信

        public Entity.ClientResult SendMsg(string templateKey, string telePhone, params string[] parameters)
        {
            Entity.ClientResult ret = new Entity.ClientResult() { Success = true };
            try
            {
                if (SMS != null)
                {
                    KeyValueModel info = smsConfigInfo.SMSTemplateList.FirstOrDefault(p => p.Key == templateKey);
                    if (info != null)
                    {
                        Dictionary<string, object> retData = SMS.SendTemplateSMS(telePhone, info.Value, parameters);
                        if (retData["statusMsg"].ToString() != "成功")
                        {
                            string msg = retData["statusMsg"].ToString();
                            ElincloudServer.Common.Utils.LogUtil.RecordErrorLog(msg);
                            //return false;
                            ret.Success = false;
                            ret.Message = msg;
                        }
                        else
                        {
                            ret.Success = true;
                        }
                    }
                    else
                    {
                        ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("未找到[" + templateKey + "]的短信模板信息!");
                        //return false;
                        ret.Success = false;
                        ret.Message = "发送短信失败，未找到对应的模板信息!";
                    }
                }
                else
                {
                    ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("初始化短信服务失败!");
                    //return false;
                    ret.Success = false;
                    ret.Message = "初始化短信服务失败!";
                }
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("发短信异常！", ex);
                //return false;
                ret.Success = false;
                ret.Message = ex.Message;
            }
            return ret;
        }

        #endregion

        public void SetSMSParameter(SMSConfigInfo config)
        {
            if (sms == null)
            {
                sms = new CCPRestSDK();
                sms.enabeLog(true);
            }

            string url = config.ServerUrl;
            string prot = config.Port;
            string accountSID = config.AccountSID;
            string accountToken = config.AccountToken;
            string appID = config.AppID;

            if (string.IsNullOrWhiteSpace(url))
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("短信服务URL未配置！");
                sms = null;
            }
            if (string.IsNullOrWhiteSpace(prot))
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("短信服务prot未配置！");
                sms = null;
            }
            if (string.IsNullOrWhiteSpace(accountSID))
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("短信服务accountSID未配置！");
                sms = null;
            }
            if (string.IsNullOrWhiteSpace(accountToken))
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("短信服务accountToken未配置！");
                sms = null;
            }
            if (string.IsNullOrWhiteSpace(appID))
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("短信服务appID未配置！");
                sms = null;
            }

            if (sms != null)
            {
                bool success = sms.init(url, prot);
                if (success)
                {
                    sms.setAccount(accountSID, accountToken);
                    sms.setAppId(appID);
                }
                else
                {
                    sms = null;
                }
            }
            smsConfigInfo = config;
        }

        private CCPRestSDK sms = null;
        private CCPRestSDK SMS
        {
            get
            {
                return sms;
            }
        }

    }
}
