using ElincloudServer.Common;
using ElincloudServer.CommonBLService;
using ElincloudServer.Entity;

using static Azure.Core.HttpHeader;

namespace ElincloudAISliceServer.ArchiveCompletenessService
{
    public class ArchiveCompletenessConfigUtil : WorkerConfigUtilBase
    {
        public static ArchiveCompletenessConfigUtil Instance = new ArchiveCompletenessConfigUtil();

        /// <summary>
        /// 医院信息
        /// </summary>
        public List<HospitalInfo> HospitalInfos { get; private set; }

        /// <summary>
        /// 文件夹地址
        /// </summary>
        public string FilePath { get; private set; }

        public void Init()
        {
            try
            {
                WorkerName = "序列归档状态更新服务";
                DeleteWorkerRunStatusLogContent();
                HospitalInfos = CommonBL.GetHospitalInfos();

                ServiceCommonConfigInfo folderConfig = CommonBL.GetServiceCommonConfigInfo();
                //缓存文件夹路径
                FilePath = folderConfig.FileStoragePath;
                if (!System.IO.Directory.Exists(FilePath))
                {
                    throw new FileNotFoundException("未配置影像存放地址");
                }

                InitSuccess = true;
            }
            catch (Exception ex)
            {
                PrintLogAction("初始化异常！", true, null, ex);
                InitSuccess = false;
            }
        }
    }
}
