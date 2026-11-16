namespace ElincloudAISliceServer.Entity
{
    /// <summary>
    /// 切割记录
    /// </summary>
    public class AIFilmSliceRecordInfo
    {
        /// <summary>
        /// 主键ID
        /// </summary>
        public string ID { get; set; } = string.Empty;

        /// <summary>
        /// 切割影像状态 
        /// </summary>
        public SliceImageStatus SliceImageStatus { get; set; } = SliceImageStatus.正在切割;

        /// <summary>
        /// 错误消息 
        /// </summary>
        public string Message { get; set; } = string.Empty;

        /// <summary>
        /// 切割结果路径 
        /// </summary>
        public string SlicePath { get; set; } = string.Empty;

        /// <summary>
        /// 是否已打印
        /// </summary>
        public bool IsPrint { get; set; } = false;


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
        /// 影像检查部位Tag
        /// </summary>
        public string BodyPartExamined { get; set; } = string.Empty;

        /// <summary>
        /// 序列号
        /// </summary>
        public string SeriesNumber { get; set; } = string.Empty;

        /// <summary>
        /// 序列描述
        /// </summary>
        public string SeriesDescription { get; set; } = string.Empty;

        /// <summary>
        /// 影像数量
        /// </summary>
        public int SeriesImageCount { get; set; } =0;

        /// <summary>
        /// 层厚
        /// </summary>
        public string SliceThickness { get; set; } = string.Empty;

        /// <summary>
        /// 是否接收成功
        /// </summary>
        public bool ReceiveCompleted { get; set; } = false;

        /// <summary>
        /// 接收完成时间
        /// </summary>
        public DateTime ReceiveCompletedDate { get; set; }

        /// <summary>
        /// 设备 
        /// </summary>
        public string MachineKey { get; set; } = string.Empty;



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
        public string ScanThickness { get; set; } = string.Empty;
    }
}