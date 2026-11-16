namespace ElincloudAISliceServer.Entity
{
    /// <summary>
    /// 切割结果
    /// </summary>
    public class AIFilmSliceResultInfo
    {
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
        /// 切片方案配置
        /// </summary>
        public SlicePlanConfigInfo SlicePlanConfigInfo { get; set; } = new SlicePlanConfigInfo();

        /// <summary>
        /// 区域
        /// </summary>
        public SliceRegions SliceRegion { get; set; } = SliceRegions.Default;

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
        /// 影像列表
        /// </summary>
        public List<string> ImageList { get; set; } = new List<string>();
    }
}