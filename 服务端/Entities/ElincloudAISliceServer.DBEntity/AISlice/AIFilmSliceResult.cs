using ElincloudServer.DBEntity;

namespace ElincloudAISliceServer.DBEntity
{
    /// <summary>
    /// 切割结果
    /// </summary>
    public class AIFilmSliceResult : BaseDBEntity
    {
        public const string _PushStatus = "PushStatus";
        public const string _PushMessage = "PushMessage";
        public const string _PushDate = "PushDate";


        /// <summary>
        /// 主键ID
        /// </summary>
        public string ID { get; set; } = string.Empty;

        /// <summary>
        /// 切割记录ID
        /// </summary>
        public string AIFilmSliceRecordID { get; set; } = string.Empty;

        /// <summary>
        /// 切割方案ID
        /// </summary>
        public string SlicePlanConfigID { get; set; } = string.Empty;

        /// <summary>
        /// 区域
        /// </summary>
        public int SliceRegion { get; set; } = 0;

        /// <summary>
        /// 实际层厚(mm)
        /// </summary>
        public double SliceThickness { get; set; } = 0;

        /// <summary>
        /// 实际层间距(mm)
        /// </summary>
        public double SliceSpacing { get; set; } = 0;

        /// <summary>
        /// 实际切割影像数量
        /// </summary>
        public int SliceImageCount { get; set; } = 0;

        /// <summary>
        /// 影像结果路径
        /// </summary>
        public string ImagePath { get; set; } = string.Empty;

        /// <summary>
        /// 推送状态
        /// </summary>
        public int PushStatus { get; set; } = 0;

        /// <summary>
        /// 推送日志
        /// </summary>
        public string PushMessage { get; set; } = string.Empty;

        /// <summary>
        /// 推送日期
        /// </summary>
        public DateTime PushDate { get; set; } = new DateTime(1900, 1, 1);
    }
}