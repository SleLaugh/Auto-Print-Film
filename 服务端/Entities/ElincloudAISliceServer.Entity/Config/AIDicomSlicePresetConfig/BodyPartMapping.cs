namespace ElincloudAISliceServer.Entity
{
    /// <summary>
    /// 检查部位映射关系
    /// </summary>
    public class BodyPartMapping
    {
        /// <summary>
        /// 序列检查部位tag值
        /// </summary>
        public string SeriesStudyPartTag { get; set; }

        /// <summary>
        /// 检查部位名称
        /// </summary>
        public string StudyPartName { get; set; }

        // 新增：无参构造函数（防止反序列化等场景报错）
        public BodyPartMapping() { }

        // 新增：带参数的构造函数
        public BodyPartMapping(string seriesStudyPartTag, string studyPartName)
        {
            SeriesStudyPartTag = seriesStudyPartTag;
            StudyPartName = studyPartName;
        }
    }
}
