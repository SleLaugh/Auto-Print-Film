using ElincloudServer.Entity;
using System.Net.Mail;
using System.Net;

namespace ElincloudServer.Common
{
    public class EmailUtil
    {
        /// <summary>
        /// 邮件模版list
        /// </summary>
        /// <remarks>
        /// sle 2025年2月19日13:59:17
        /// </remarks>
        public readonly List<KeyValueModel> EmailTemplateList = new List<KeyValueModel>()
         {
             new KeyValueModel(){Key = "测试邮件",Value="邮件发送时间(yyyy-MM-dd HH:mm:ss)" },
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
             new KeyValueModel(){Key = "影像序列文件邮件",Value="医院名称，下载日期，检查号，有效期，下载链接" },
             new KeyValueModel(){Key = "归档预警",Value="云主机名称，积压上限" },
             new KeyValueModel(){Key = "前置机运行状态预警",Value="医院名称，几个服务，多久不运行" }
         };

        //public string DcmFileTooMuchStr = "【易联云影】云影像用户[{0}]前置机积压未上传原始影像超过[{1}]张，请及时处理。";
        //public string DataSynchronizationWarningStr = "【易联云影】云影像用户[{0}]前置机[{1}]数据同步比当前时间慢[{2}]分钟，请及时处理";
        //public string DiskWarningStr = "【易联云影】[{0}]前置机的磁盘空间不足{1}，请及时处理。";
        //public string CpuWarningStr = "【易联云影】[{0}]前置机的CPU使用率长时间超过[{1}]，请及时处理。";

        private EmailConfigInfo emailConfigInfo = new EmailConfigInfo();

        public void SetEmailParameter(EmailConfigInfo config)
        {
            emailConfigInfo.SMTPAddress = config.SMTPAddress;
            emailConfigInfo.SMTPPort = config.SMTPPort;
            emailConfigInfo.SendEmail = config.SendEmail;
            emailConfigInfo.SendEmailAuthorizationCode = config.SendEmailAuthorizationCode;
            emailConfigInfo.EmailTemplateList = config.EmailTemplateList;

            if (string.IsNullOrEmpty(emailConfigInfo.SendEmailAuthorizationCode))
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("邮件服务地址不能为空！");
            }
            if (emailConfigInfo.SMTPPort == 0)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("邮件服务端口未配置！");
            }
            if (string.IsNullOrWhiteSpace(emailConfigInfo.SendEmail))
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("发送邮箱账号未配置！");
            }
            if (string.IsNullOrWhiteSpace(emailConfigInfo.SendEmailAuthorizationCode))
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("发送邮箱授权码未配置！");
            }
        }

        /// <summary>
        /// 发送邮件
        /// </summary>
        /// <param name="templateKey">模版Key（同时是邮件主标题）</param>
        /// <param name="recipientEmail">接收方邮件地址</param>
        /// <param name="parameters">邮件参数</param>
        /// <returns></returns>
        public ClientResult SendEmail(string templateKey, List<string> recipientEmails, params string[] parameters)
        {
            ClientResult ret = new Entity.ClientResult() { Success = false };
            try
            {
                KeyValueModel template = emailConfigInfo.EmailTemplateList.FirstOrDefault(i => i.Key == templateKey);
                if (template == null)
                {
                    ret.Success = false;
                    ret.Message = "发送邮件失败，未找到对应的模板信息";
                    return ret;
                }

                using (MailMessage mail = new MailMessage())
                {
                    mail.From = new MailAddress(emailConfigInfo.SendEmail);

                    foreach (string email in recipientEmails)
                    {
                        mail.To.Add(new MailAddress(email));
                    }

                    mail.Subject = templateKey;
                    mail.Body = string.Format(template.Value, parameters);
                    mail.IsBodyHtml = true;

                    using (SmtpClient client = new SmtpClient(emailConfigInfo.SMTPAddress))
                    {
                        client.Port = emailConfigInfo.SMTPPort;
                        client.Credentials = new NetworkCredential(emailConfigInfo.SendEmail, emailConfigInfo.SendEmailAuthorizationCode);
                        client.EnableSsl = true;
                        client.Send(mail);
                    }
                }

                ret.Success = true;
                ret.Message = "邮件发送成功";
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("发邮件异常！", ex);
                ret.Success = false;
                ret.Message = ex.Message;
            }
            return ret;
        }
    }
}
