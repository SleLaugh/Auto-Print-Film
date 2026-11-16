using ElincloudServer.BLServices;
using ElincloudServer.Entity;
using Microsoft.AspNetCore.Mvc;

namespace ElincloudServer.Controllers
{
    public class IMController : BaseController
    {
        #region 用户登录

        public JsonResult Login(string loginName, string password, ControllerParameter<bool> isAdmin)//string loginname, string password,
        {
            return Excute(new Func<Object>(() =>
            {
                ClientResult ret = IMServiceBL.Instance.Login(loginName, password, isAdmin.Value);
                return ret;
            }), "E00000220");
        }

        public JsonResult Logout()
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                IMCenterUtil.IMCenter.UpdateLogoutTokens(pi);
                Service.UpdateLogoutQueueMachines(pi.LoginName);
            }), "E00000221");
        }

        public JsonResult GetLoginUserInfo()
        {
            return Excute<Dictionary<string, IMClientInfo>>(new Func<PermissionInfo, Dictionary<string, IMClientInfo>>((PermissionInfo pi) =>
            {
                return IMCenterUtil.IMCenter.DicClients;
            }), "E00000571");
        }

        #endregion
    }
}
