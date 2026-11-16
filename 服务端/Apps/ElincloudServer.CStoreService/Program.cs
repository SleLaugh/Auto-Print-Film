using ElincloudServer.Common;
using ElincloudServer.Entity;

namespace ElincloudServer.CStoreService
{
    public class Program
    {
        public static void Main(string[] args)
        {
            string? appName = typeof(Program).Namespace;
            if (!Utils.ProcessUtil.CheckProcess(appName))
            {
                ServiceInfo serviceInfo = CommonBLService.CommonBL.GetServerInfoByServerENName(appName);
                Utils.ProcessUtil.SetProcessNameForLinux(serviceInfo.ServiceShortName);
                IHost host = Host.CreateDefaultBuilder(args)
                .ConfigureAppConfiguration(config =>
                {
                    config.Sources.Clear();
                })
                .AddSignalRHost(serviceInfo.ServiceRunLogPort)
                .ConfigureServices(services =>
                {
                    services.AddHostedService<CStoreWorker>();
                })
                .Build();
                //初始化收图服务参数
                CStoreConfigUtil.Instance.Init();

                if (CStoreConfigUtil.Instance.InitSuccess)
                {
                    host.Run();
                }
            }
        }
    }
}