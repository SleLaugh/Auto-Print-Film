namespace ElincloudAISliceServer.DBEntity
{
    public class ExaminationImage : ElincloudServer.DBEntity.ExaminationImage
    {
        public static string _MachineKey = "MachineKey";

        public static string _BodyPartExamined = "BodyPartExamined";
        public static string _ScanThickness = "ScanThickness";
        public static string _SeriesDescription = "SeriesDescription";

        public static string _ReceiveCompletedDate = "ReceiveCompletedDate";
        public static string _ReceiveCompleted = "ReceiveCompleted";
        public const string _SliceImageStatus = "SliceImageStatus";

        /// <summary>
        /// 影像检查部位Tag
        /// </summary>
        public string BodyPartExamined { get; set; } = string.Empty;

        /// <summary>
        /// 扫描厚度
        /// </summary>
        public string ScanThickness { get; set; } = string.Empty;

        /// <summary>
        /// 序列描述
        /// </summary>
        public string SeriesDescription { get; set; } = string.Empty;

        /// <summary>
        /// 创建日期
        /// </summary>
        public DateTime CreateDate { get; set; } = DateTime.Now;

        /// <summary>
        /// 影像接收完成
        /// </summary>
        public bool ReceiveCompleted { get; set; } = false;

        /// <summary>
        /// 接收完成时间
        /// </summary>
        public DateTime ReceiveCompletedDate { get; set; } = Convert.ToDateTime("1900-01-01 00:00:00");

        /// <summary>
        /// 切割影像状态 
        /// </summary>
        public int SliceImageStatus { get; set; } = 0;

        /// <summary>
        /// 设备 
        /// </summary>
        public string MachineKey { get; set; } = string.Empty;
    }
}
