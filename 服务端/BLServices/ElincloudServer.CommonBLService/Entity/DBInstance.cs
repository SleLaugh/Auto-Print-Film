using ElincloudServer.DBService;

namespace ElincloudServer.CommonBLService
{
    internal class DBInstance
    {
        public string Key { get; set; }
        public SqlAccess Instance { get; set; }
    }
}
