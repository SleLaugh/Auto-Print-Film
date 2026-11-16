using ElincloudAISliceServer.Entity;
using ElincloudServer.Common;
using ElincloudServer.CommonBLService;
using ElincloudServer.Entity;

namespace ElincloudAISliceServer.AISliceDicomService
{
    public class AISliceDicomServiceUtil : WorkerConfigUtilBase
    {
        public static AISliceDicomServiceUtil Instance = new AISliceDicomServiceUtil();

        /// <summary>
        /// 每30秒执行一下
        /// </summary>
        public int ExecuteWaitMinute { get; private set; }

        /// <summary>
        /// 医院信息
        /// </summary>
        public List<HospitalInfo> HospitalInfos { get; private set; }

        /// <summary>
        /// AI影像切割配置
        /// </summary>
        public AIDicomSliceConfigInfo AIFilmSliceConfigInfo { get; private set; }

        /// <summary>
        /// 系统配置
        /// </summary>
        public ServiceCommonConfigInfo ServiceCommonConfigInfo { get; private set; }

        public void Init()
        {
            try
            {
                WorkerName = "智能排片切割影像服务";
                DeleteWorkerRunStatusLogContent();
                ExecuteWaitMinute = 1000 * 20;
                HospitalInfos = CommonBL.GetHospitalInfos();
                AIFilmSliceConfigInfo = CommonBL.GetAIDicomSliceConfigInfo();
                ServiceCommonConfigInfo = CommonBL.GetServiceCommonConfigInfo();

                if (string.IsNullOrWhiteSpace(AIFilmSliceConfigInfo.ExecutablePath))
                {
                    throw new InvalidOperationException("未配置智能影像切割可执行程序路径");
                }

                if (!File.Exists(AIFilmSliceConfigInfo.ExecutablePath))
                {
                    throw new FileNotFoundException("找不到智能影像切割可执行程序", AIFilmSliceConfigInfo.ExecutablePath);
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
