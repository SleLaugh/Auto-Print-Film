using ElincloudServer.Common;
using ElincloudServer.ConfigBLService;
using ElincloudServer.Entity;
using DicomMachineConfigInfo = ElincloudAISliceServer.Entity.DicomMachineConfigInfo;
using DicomMachineInfo = ElincloudAISliceServer.Entity.DicomMachineInfo;

namespace ElincloudServer.CStoreService
{
    internal class CStoreConfigUtil : WorkerConfigUtilBase
    {
        public static CStoreConfigUtil Instance = new CStoreConfigUtil();

        private ConfigBL configBL = null;
        private ConfigBL ConfigBLInstance
        {
            get
            {
                if (configBL == null)
                {
                    configBL = new ConfigBL();
                }
                return configBL;
            }
        }

        public Dicom.Network.Client.DicomClient DCMTransferClient { get; private set; }

        /// <summary>
        /// 服务IP
        /// </summary>
        public string IP { get; private set; }

        /// <summary>
        /// 服务端口
        /// </summary>
        public int Port { get; private set; }

        /// <summary>
        /// 服务AE
        /// </summary>
        public string ServerAE { get; private set; }

        /// <summary>
        /// 设备收图配置
        /// </summary>
        public CStoreConfigInfo Config { get; private set; }

        public List<CStoreConfigInfo> CStoreConfigInfos { get; private set; }

        /// <summary>
        /// 归档监听路径
        /// </summary>
        public string ImageSourcePath { get; private set; }
        /// <summary>
        /// 接收文件缓存目录
        /// </summary>
        public string RecieveFilePath { get; private set; }

        public bool DicomImageTransferEnabled { get; private set; }

        public Dictionary<string, DicomMachineInfo> DicomMachineDic = new Dictionary<string, DicomMachineInfo>();

        public int ArchiveProssessCount { get; private set; }

        public void Init()
        {
            #region 初始化配置

            try
            {
                System.Threading.ThreadPool.SetMinThreads(500, 500);
                System.Threading.ThreadPool.SetMaxThreads(1000, 1000);

                WorkerName = "图像接收服务";
                DeleteWorkerRunStatusLogContent();
                ServiceCommonConfigInfo folderConfig = ConfigBLInstance.GetServiceCommonConfigInfo();
                //归档监听路径
                ImageSourcePath = folderConfig.ArchivePath;
                if (!System.IO.Directory.Exists(ImageSourcePath))
                {
                    Common.Utils.FileUtil.CreateDirectory(ImageSourcePath);
                }
                //接收文件缓存目录
                RecieveFilePath = ElincloudServer.Common.Utils.StringUtil.Combine(AppDomain.CurrentDomain.BaseDirectory, ElincloudServer.Common.ApplicationEx.CStoreCacheFolder);
                if (!System.IO.Directory.Exists(RecieveFilePath))
                {
                    Common.Utils.FileUtil.CreateDirectory(RecieveFilePath);
                }
                List<ArchiveConfigInfo> archiveConfigs = configBL.GetArchiveConfigInfos();
                ArchiveProssessCount = archiveConfigs.FirstOrDefault().ArchiveProssessCount;

                List<CStoreConfigInfo> configs = ConfigBLInstance.GetCStoreConfigInfos();
                if (configs.Count > 0)
                {
                    CStoreConfigInfos = configs;
                    //在服务器启动该服务只使用第一个配置
                    CStoreConfigInfo config = configs[0];
                    IP = config.IP;
                    Port = config.Port;
                    ServerAE = config.AE;
                    Config = config;
                    DicomImageTransferEnabled = config.DicomImageTransferEnabled;
                    if (DicomImageTransferEnabled)
                    {
                        DCMTransferClient = new Dicom.Network.Client.DicomClient(config.DicomImageTransferIP, config.DicomImageTransferPort, false, config.AE, config.DicomImageTransferServerAE);
                        DCMTransferClient.NegotiateAsyncOps();
                    }
                    List<DicomMachineConfigInfo> dicomMachineConfigInfos = ConfigBLInstance.GetDicomMachineInfos();
                    foreach (var dicomMachineConfigInfo in dicomMachineConfigInfos)
                    {
                        foreach (var machine in dicomMachineConfigInfo.DicomMachines)
                        {
                            DicomMachineDic.Add(GetDicomMachineKey(machine.IP, machine.AE), machine);
                        }
                    }
                }
                string ip = string.Empty;
                if (IP == "127.0.0.1")
                {
                    List<string> ips = ElincloudServer.Common.Utils.StringUtil.GetLocalIPV4s();
                    ip = string.Join("，  ", ips.ToArray());
                }
                else
                {
                    ip = IP;
                }
                ServerRunParmInfo = string.Format("服务IP：{0} 服务AE：{1} 端口：{2}", ip, Config.AE, Config.Port);
                PrintStatusLogAction(ServerRunParmInfo);
                InitSuccess = true;
            }
            catch (Exception ex)
            {
                PrintLogAction("初始化异常！", true, null, ex);
                InitSuccess = false;
            }
            #endregion
        }
    }
}
