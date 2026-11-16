namespace ElincloudAISliceServer.Entity
{
    /// <summary>
    /// 切割规则
    /// </summary>
    public class SliceRuleConfigInfo
    {
        /// <summary>
        /// 带参数的构造函数
        /// </summary>
        /// <param name="studyType">检查类型</param>
        /// <param name="slicePart">切割部位</param>
        /// <param name="plane">状位</param>
        /// <param name="sliceType">切割方式</param>
        /// <param name="regions">切割区域</param>
        public SliceRuleConfigInfo(string studyType, string slicePart, PlaneEnums plane, SliceTypes sliceType, List<SliceRegions> regions)
        {
            StudyType = studyType;
            SlicePart = slicePart;
            Plane = plane;
            SliceType = sliceType;
            Regions = regions ?? new List<SliceRegions>() { };
        }

        /// <summary>
        /// 无参数构造函数（保留默认值）
        /// </summary>
        public SliceRuleConfigInfo()
        {
        }

        /// <summary>
        /// 检查类型
        /// </summary>
        public string StudyType { get; set; }

        /// <summary>
        /// 切割部位
        /// </summary>
        public string SlicePart { get; set; }

        /// <summary>
        /// 状位
        /// </summary>
        public PlaneEnums Plane { get; set; }

        /// <summary>
        /// 切割方式
        /// </summary>
        public SliceTypes SliceType { get; set; }

        /// <summary>
        /// 切割区域
        /// </summary>
        public List<SliceRegions> Regions { get; set; } = new List<SliceRegions>() { };
    }
}
