using ElincloudServer.Common;
using ElincloudServer.Entity;

namespace ElincloudAISliceServer.AISliceDicomService
{
    public class Program
    {
        static void Main(string[] args)
        {
            string? appName = typeof(Program).Namespace;
            if (!Utils.ProcessUtil.CheckProcess(appName))
            {
                ServiceInfo serviceInfo = ElincloudServer.CommonBLService.CommonBL.GetServerInfoByServerENName(appName);
                Utils.ProcessUtil.SetProcessNameForLinux(serviceInfo.ServiceShortName);
                IHost host = Host.CreateDefaultBuilder(args)
                .ConfigureAppConfiguration(config =>
                {
                    config.Sources.Clear();
                })
                .AddSignalRHost(serviceInfo.ServiceRunLogPort)
                .ConfigureServices(services =>
                {
                    services.AddHostedService<AISliceDicomWorker>();
                })
                .Build();
                AISliceDicomServiceUtil.Instance.Init();
                if (AISliceDicomServiceUtil.Instance.InitSuccess)
                {
                    host.Run();
                }
            }
        }
    }
}
