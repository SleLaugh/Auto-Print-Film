using ElincloudServer.DBEntity;

namespace ElincloudAISliceServer.DBEntity
{
    /// <summary>
    /// 切割记录
    /// </summary>
    public class AIFilmSliceRecord : BaseDBEntity
    {
        public const string _SliceImageStatus = "SliceImageStatus";
        public const string _Message = "Message";
        public const string _ProcessID = "ProcessID";
        public const string _StartDate = "StartDate";

        /// <summary>
        /// 主键ID
        /// </summary>
        public string ID { get; set; } = string.Empty;

        /// <summary>
        /// 切割影像状态 
        /// </summary>
        public int SliceImageStatus { get; set; } = 0;

        /// <summary>
        /// 错误消息 
        /// </summary>
        public string Message { get; set; } = string.Empty;

        /// <summary>
        /// 进程ID 
        /// </summary>
        public int ProcessID { get; set; } = 0;

        /// <summary>
        /// 启动时间
        /// </summary>
        public DateTime StartDate { get; set; } = Convert.ToDateTime("1900-01-01 00:00:00");

        /// <summary>
        /// 切割结果路径 
        /// </summary>
        public string SlicePath { get; set; } = string.Empty;


        /// <summary>
        /// 影像号
        /// </summary>
        public string PatientID { get; set; } = string.Empty;

        /// <summary>
        /// 检查号
        /// </summary>
        public string ExaminationID { get; set; } = string.Empty;

        /// <summary>
        /// SeriesInstanceUID
        /// </summary>
        public string SeriesInstanceUID { get; set; } = string.Empty;

        /// <summary>
        /// 检查类型
        /// </summary>
        public string StudyTypeName { get; set; } = string.Empty;

        /// <summary>
        /// 检查部位
        /// </summary>
        public string StudyPartName { get; set; } = string.Empty;

        /// <summary>
        /// 切割部位
        /// </summary>
        public string SlicePartName { get; set; } = string.Empty;

        /// <summary>
        /// 层厚
        /// </summary>
        public double ScanThickness { get; set; } = 0;

        /// <summary>
        /// 切割方案ID
        /// </summary>
        public string SliceConfigIDs { get; set;  }= string.Empty;

        /// <summary>
        /// 输入路径
        /// </summary>
        public string InputDir { get; set; } = string.Empty;

        /// <summary>
        /// 缓存文件路径
        /// </summary>
        public string CacheDir { get; set; } = string.Empty;

        /// <summary>
        /// 输出路径
        /// </summary>
        public string OutputDir { get; set; } = string.Empty;

        /// <summary>
        /// 状位参数
        /// </summary>
        public string PlaneConfigJson { get; set; } = string.Empty;
    }
}