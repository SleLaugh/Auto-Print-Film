using ElincloudAISliceServer.Entity;
using ElincloudServer.ArchiveBLService;
using ElincloudServer.Common;
using ElincloudServer.CommonBLService;
using ElincloudServer.ConfigBLService;
using ElincloudServer.Entity;
using DCMImageInfo = ElincloudAISliceServer.Entity.DCMImageInfo;

namespace ElincloudServer.ArchiveService
{
    internal class ArchiveConfigUtil : WorkerConfigUtilBase
    {
        public static ArchiveConfigUtil Instance = new ArchiveConfigUtil();

        private static readonly object archiveFileTimeLockObj = new object();
        private static readonly Dictionary<string, DateTime> archiveFileTimes = new Dictionary<string, DateTime>();

        public static void UpdateArchiveFileTime(string examinationID, string patientID, DateTime archiveTime)
        {
            if (string.IsNullOrWhiteSpace(examinationID) || string.IsNullOrWhiteSpace(patientID))
            {
                return;
            }
            lock (archiveFileTimeLockObj)
            {
                archiveFileTimes[examinationID + "|" + patientID] = archiveTime;
            }
        }

        public static List<KeyValuePair<string, DateTime>> GetExpiredArchiveFileTimes(TimeSpan waitTime)
        {
            lock (archiveFileTimeLockObj)
            {
                DateTime now = DateTime.Now;
                return archiveFileTimes.Where(p => (now - p.Value) > waitTime).ToList();
            }
        }

        public static void RemoveArchiveFileTime(string key)
        {
            lock (archiveFileTimeLockObj)
            {
                archiveFileTimes.Remove(key);
            }
        }

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

        private ArchiveBL archiveBL = null;
        public ArchiveBL ArchiveBLInstance
        {
            get
            {
                if (archiveBL == null)
                {
                    archiveBL = new ArchiveBL();
                }
                return archiveBL;
            }
        }

        private bool isEnableArchiveCache = false;
        /// <summary>
        /// 是否启用缓存
        /// </summary>
        public bool IsEnableArchiveCache
        {
            get { return isEnableArchiveCache; }
            set { isEnableArchiveCache = value; }
        }

        private int archiveCacheFileCount = 50;
        /// <summary>
        /// 缓存文件数量
        /// </summary>
        public int ArchiveCacheFileCount
        {
            get { return archiveCacheFileCount; }
            set { archiveCacheFileCount = value; }
        }


        /// <summary>
        /// 归档监听路径
        /// </summary>
        public string ImageSourcePath { get; private set; }

        /// <summary>
        /// 归档监听Error路径
        /// </summary>
        public string ImageSourceErrorPath { get; private set; }

        /// <summary>
        /// 归档图像保存路径
        /// </summary>
        public string ImageTargetPath { get; private set; }

        /// <summary>
        /// 归档图像保存文件夹
        /// </summary>
        public string ImageFolder { get; private set; }

        /// <summary>
        /// 归档数据缓存路径
        /// </summary>
        public string ImageInfoPath { get; private set; }

        /// <summary>
        /// 归档数据缓存Error路径
        /// </summary>
        public string ImageInfoErrorPath { get; private set; }

        private bool isEnableMultArchiveFolder = false;
        /// <summary>
        /// 是否启用多归档路径
        /// </summary>
        public bool IsEnableMultArchiveFolder
        {
            get { return isEnableMultArchiveFolder; }
            set { isEnableMultArchiveFolder = value; }
        }

        private List<ArchiveFolderConfigInfo> archiveFolderConfigInfos = new List<ArchiveFolderConfigInfo>();
        /// <summary>
        /// 归档目录配置
        /// </summary>
        public List<ArchiveFolderConfigInfo> ArchiveFolderConfigInfos
        {
            get { return archiveFolderConfigInfos; }
            set { archiveFolderConfigInfos = value; }
        }

        internal MultiDictionary<string, DCMImageInfo> ArchiveDetails = new MultiDictionary<string, DCMImageInfo>();

        private Dictionary<string, ArchiveConfigInfo> archiveConfigDics = new Dictionary<string, ArchiveConfigInfo>();
        public ArchiveConfigInfo GetArchiveConfig(string hospitalCode)
        {
            if (!archiveConfigDics.ContainsKey(hospitalCode))
            {
                return null;
            }
            else
            {
                return archiveConfigDics[hospitalCode];
            }
        }

        private MultiDictionary<string, CheckInfo> studyTypeDics = new MultiDictionary<string, CheckInfo>();
        public CheckInfo GetStudyType(string hospitalCode, string name)
        {
            if (!studyTypeDics.ContainsKey(hospitalCode))
            {
                List<CheckInfo> infos = CommonBL.GetCheckInfos();
                if (infos.Count > 0)
                {
                    foreach (var item in infos)
                    {
                        studyTypeDics.Add(hospitalCode, item);
                    }
                }
                else
                {
                    //1.如果没有获取到医院的检查类型，这里直接添加一个默认保证下一个序列过来不重复查数据库
                    //2.在归档接口（FinishArchive）中如果没有查询到Modality对应的检查类型信息则不会修改检查状态，
                    //  但是得保证云ExaminationImage可以正常写入，因为在云影像场景下要使用ExaminationImage数据
                    studyTypeDics.Add(hospitalCode, new CheckInfo());
                }
            }
            return studyTypeDics[hospitalCode].FirstOrDefault(p => p.Name == name);
        }

        public void Init(int instanceID)
        {
            #region 初始化配置

            try
            {
                WorkerName = "数据归档服务";
                ServiceCommonConfigInfo folderConfig = CommonBL.GetServiceCommonConfigInfo();
                if (instanceID == 1) //服务首次启动初始化参数
                {
                    DeleteWorkerRunStatusLogContent();
                }

                //首次初始化成功后，服务运行的参数初始化时创建对应的路径信息

                //归档监听路径
                ImageSourcePath = ElincloudServer.Common.Utils.StringUtil.Combine(folderConfig.ArchivePath, instanceID.ToString());
                if (!System.IO.Directory.Exists(ImageSourcePath))
                {
                    Common.Utils.FileUtil.CreateDirectory(ImageSourcePath);
                }
                //归档监听Error路径
                ImageSourceErrorPath = ElincloudServer.Common.Utils.StringUtil.Combine(ImageSourcePath, "ErrorFiles");
                if (!System.IO.Directory.Exists(ImageSourceErrorPath))
                {
                    Common.Utils.FileUtil.CreateDirectory(ImageSourceErrorPath);
                }
                ImageTargetPath = folderConfig.FileStoragePath;
                //归档图像保存路径    
                if (!System.IO.Directory.Exists(ImageTargetPath))
                {
                    Common.Utils.FileUtil.CreateDirectory(ImageTargetPath);
                }
                //归档图像保存文件夹
                ImageFolder = folderConfig.DicomImageStorageFolder;
                string imagePath = ElincloudServer.Common.Utils.StringUtil.Combine(ImageTargetPath, ImageFolder);
                if (!System.IO.Directory.Exists(imagePath))
                {
                    Common.Utils.FileUtil.CreateDirectory(imagePath);
                }
                //归档数据缓存路径
                ImageInfoPath = ElincloudServer.Common.Utils.StringUtil.Combine(folderConfig.ArchivePath, ApplicationEx.ArchiveDataFolder); ;
                if (!System.IO.Directory.Exists(ImageInfoPath))
                {
                    Common.Utils.FileUtil.CreateDirectory(ImageInfoPath);
                }

                //归档数据缓存Error路径
                ImageInfoErrorPath = ElincloudServer.Common.Utils.StringUtil.Combine(ImageInfoPath, "ErrorFiles"); ;
                if (!System.IO.Directory.Exists(ImageInfoErrorPath))
                {
                    Common.Utils.FileUtil.CreateDirectory(ImageInfoErrorPath);
                }
                //归档配置
                List<ArchiveConfigInfo> archiveConfigs = ConfigBLInstance.GetArchiveConfigInfos();
                if (archiveConfigs.Count > 0)
                {
                    ArchiveFolderConfigInfos = archiveConfigs[0].ArchiveFolderConfigInfos;
                    IsEnableMultArchiveFolder= archiveConfigs[0].IsEnableMultArchiveFolder;
                    foreach (var config in archiveConfigs)
                    {
                        archiveConfigDics.Add(config.Hospital.Code, config);
                    }
                    IsEnableArchiveCache = archiveConfigs.FirstOrDefault().IsEnableArchiveCache;
                    archiveCacheFileCount = archiveConfigs.FirstOrDefault().ArchiveCacheFileCount;
                    if (IsEnableMultArchiveFolder)
                    {
                        foreach (var item in ArchiveFolderConfigInfos)
                        {
                            Common.Utils.FileUtil.CreateDirectory(item.FolderName);
                        }
                    }
                }

                //检查类型缓存
                InitSuccess = true;
            }
            catch (Exception ex)
            {
                PrintLogAction("初始化异常!", true, null, ex);
                InitSuccess = false;
            }
            #endregion
        }
    }
}
