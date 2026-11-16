using Dicom.Network;
using Dicom;

namespace ElincloudServer.CStoreService
{
    public class CStoreWorker : BackgroundService
    {
        private readonly ILogger<CStoreWorker> _logger;
        private IDicomServer server = null;

        public CStoreWorker(ILogger<CStoreWorker> logger)
        {
            _logger = logger;
        }

        public override Task StartAsync(CancellationToken cancellationToken)
        {
            if (CStoreConfigUtil.Instance.InitSuccess)
            {
                collectTimer = new Timer(CollectTimer_Tick, null, TimeSpan.Zero, TimeSpan.FromMilliseconds(300000));//5分钟进行一次垃圾回收

                return base.StartAsync(cancellationToken);
            }
            else
            {
                return Task.CompletedTask;
            }
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            StartService(true);
        }

        private void StartService(bool log = false)
        {
            // preload dictionary to prevent timeouts
            var dict = DicomDictionary.Default;
            // start DICOM server 
            if (CStoreConfigUtil.Instance.IP == "127.0.0.1")
            {
                server = DicomServer.Create<CStoreSCP>(CStoreConfigUtil.Instance.Port);
            }
            else
            {
                server = DicomServer.Create<CStoreSCP>(CStoreConfigUtil.Instance.IP, CStoreConfigUtil.Instance.Port);
            }

            if (server.IsListening)
            {
                CStoreConfigUtil.Instance.PrintStatusLogAction($"服务启动成功！" + CStoreConfigUtil.Instance.ServerRunParmInfo);
                //测试cstore服务是否能连通 如果不通则重启cstore 下面这个要放到线程里面不然打出的日志格式不对 
                //Common.App.ThreadUtil.StartThread(() =>
                //{
                cstoreServerConnectTest(server);
                //});
            }
            else
            {
                CStoreConfigUtil.Instance.PrintLogAction($"启动失败！[{server.Exception.Message}]", true);
            }
        }

        bool connected = false;
        private void cstoreServerConnectTest(IDicomServer server)
        {
            string msg = $"测试连通状态";
            CStoreConfigUtil.Instance.PrintLogAction($"正在{msg}...");
            var client = new Dicom.Network.Client.DicomClient(CStoreConfigUtil.Instance.IP, CStoreConfigUtil.Instance.Port, false, "TestSCU", CStoreConfigUtil.Instance.ServerAE, 1000);
            client.AddRequestAsync(new DicomCEchoRequest
            {
                OnResponseReceived = (rq, rsp) =>
                {
                    if (!connected)
                    {
                        connected = rsp.Status == DicomStatus.Success;
                        CStoreConfigUtil.Instance.PrintLogAction(msg + (connected ? "成功！" : "失败！"));
                    }
                },
            }).ConfigureAwait(false);
            Task task = client.SendAsync();
            try
            {
                task.Wait();
                if (!connected)
                {
                    CStoreConfigUtil.Instance.PrintLogAction($"失败！");
                    if (server.IsListening)
                    {
                        server.Dispose();
                    }
                    StartService();
                }
            }
            catch (Exception ex)
            {
                if (ex.InnerException != null)
                {
                    CStoreConfigUtil.Instance.PrintLogAction($"失败！[{ex.InnerException.Message}]");
                }
            }
        }

        private Timer collectTimer = null;
        private void CollectTimer_Tick(object? state)
        {
            GC.Collect();
        }
    }
}
