
using ElincloudServer.Common;
using ElincloudServer.ConfigBLService;
using ElincloudServer.DBEntity;
using ElincloudServer.Entity;

namespace ElincloudServer.ArchiveService
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
                //初始化归档服务参数
                ArchiveConfigUtil.Instance.Init(1);
                if (ArchiveConfigUtil.Instance.InitSuccess)
                {
                    IHost host = Host.CreateDefaultBuilder(args)
                    .ConfigureAppConfiguration(config =>
                    {
                        config.Sources.Clear();
                    })
                    .AddSignalRHost(serviceInfo.ServiceRunLogPort)
                    .ConfigureServices(services =>
                    {
                        services.AddHostedService<ArchiveDBWorker>();
                        services.AddHostedService<MoveImageWorker>();
                        services.AddHostedService<DeleteDetailWorker>();
                        services.AddHostedService<ArchiveCompletenessWorker>();
                        services.AddHostedService<ArchiveDatabaseCheckWorker>();
                        ConfigBL configBL = new ConfigBL();
                        List<ArchiveConfigInfo> archiveConfigs = configBL.GetArchiveConfigInfos();
                        if (archiveConfigs.Count > 0)
                        {
                            for (int i = 1; i <= archiveConfigs.FirstOrDefault().ArchiveProssessCount; i++)
                            {
                                int instanceId = i;
                                services.AddSingleton<IHostedService>(new ArchiveFileWorker(instanceId));
                            }
                        }
                        else
                        {
                            services.AddSingleton<IHostedService>(new ArchiveFileWorker(1));
                        }
                    })
                    .Build();
                    host.Run();
                }
            }
        }
    }
}
