using ElincloudServer.Common;
using ElincloudServer.Entity;

namespace ElincloudAISliceServer.ArchiveCompletenessService
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
                    services.AddHostedService<ArchiveCompletenessWorker>();
                })
                .Build();
                ArchiveCompletenessConfigUtil.Instance.Init();
                if (ArchiveCompletenessConfigUtil.Instance.InitSuccess)
                {
                    host.Run();
                }
            }
        }
    }
}