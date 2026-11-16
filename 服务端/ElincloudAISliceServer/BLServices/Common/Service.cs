using System.Collections.Concurrent;

namespace ElincloudServer.BLServices
{
    public class Service
    {
        //验证Url暴力请求缓存
        public static ConcurrentDictionary<string, UrlRequestCache> UrlRequests = new ConcurrentDictionary<string, UrlRequestCache>();

        #region 登录与退出时更新叫号列表

        //internal static void UpdateQueueMachines(string clientID, QueueClientTypes clientType, string hospitalID, Action logoutAction, Func<bool> heartBeatAction)
        //{
        //    //if (QueueMachines.ContainsKey(clientID))
        //    //{
        //    //    QueueMachines[clientID] = new MachineCache() { ClientID = clientID, ClientType = clientType, HospitalID = hospitalID, LogoutAction = logoutAction, HeartBeatAction = heartBeatAction };
        //    //}
        //    //else
        //    //{
        //    //    QueueMachines.Add(clientID, new MachineCache() { ClientID = clientID, ClientType = clientType, HospitalID = hospitalID, LogoutAction = logoutAction, HeartBeatAction = heartBeatAction });
        //    //}
        //}

        internal static void UpdateLogoutQueueMachines(string clientID)
        {
            //if (QueueMachines.ContainsKey(clientID))
            //{
            //    if (QueueMachines[clientID].LogoutAction != null)
            //    {
            //        QueueMachines[clientID].LogoutAction();
            //    }
            //    QueueMachines.Remove(clientID);
            //}
        }


        #endregion

        #region 当前请求Headers处理方法

        public static IHttpContextAccessor HttpContextAccessor;

        public static string GetHeaderValue(string key)
        {
            string value = string.Empty;
            var headers = HttpContextAccessor.HttpContext.Request.Headers;
            foreach (var header in headers)
            {
                if (header.Key.ToLower() == key.ToLower())
                {
                    value = header.Value;
                    break;
                }
            }
            return value;
        }

        #endregion

    }
}
