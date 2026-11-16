using ElincloudServer.Entity;

namespace ElincloudServer.BLServices
{
    public class IMClientInfo : PermissionInfo
    {
        public string Token { get; set; }
        public IIMServiceCallBack CallBack { get; set; }
        //TODO:public IContextChannel Channel { get; set; }
    }
}
