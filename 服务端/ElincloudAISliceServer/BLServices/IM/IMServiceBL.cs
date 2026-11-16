using ElincloudServer.Common;
using ElincloudServer.Entity;

namespace ElincloudServer.BLServices
{
    public class IMServiceBL
    {
        public static IMServiceBL Instance = new IMServiceBL();

        public ClientResult Login(string loginName, string password, bool isAdmin)
        {
            if (string.IsNullOrEmpty(loginName) || string.IsNullOrEmpty(password))
            {
                throw new ServiceException("用户名或密码不能为空！");
            }

            ClientResult ret = new ClientResult() { Success = false };

            try
            {
                if (IMCenterUtil.IMCenter.LoginEnabled(loginName))
                {
                    AccountInfo accountInfo = BLService.SystemBLService.Login(loginName, password, isAdmin, (int remaindays) =>
                    {
                        if (remaindays <= 30)
                        {
                            IMCenterUtil.IMCenter.SendMessage(loginName, "系统的授权使用有效期即将到期。" + Environment.NewLine + "请尽快联系管理员！", "Service", "0", SysMessages.News, "授权到期提醒");
                        }
                    });
                    ret.Success = true;
                    ret.Result = accountInfo;
                }
            }
            catch (Exception ex)
            {
                ret.Message = ex.Message;
            }
            finally
            {
                if (ret.Result == null)
                {
                    ret.Result = new UserInfo() { LoginName = loginName };
                }
                ClientResult validateRet = IMCenterUtil.IMCenter.SecurityChecked(loginName, ret.Success);
                if (!validateRet.Success)
                {
                    ret.Success = false;
                    ret.Message = validateRet.Message;
                }

                IMCenterUtil.IMCenter.RegistClient(ret);
            }
            return ret;

        }

    }
}
