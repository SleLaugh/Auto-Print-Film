namespace ElincloudAISliceServer.DBEntity
{
    /// <summary>
    /// 智能切割影像配置
    /// </summary>
    public class AIDicomSliceConfig
    {
        public static string _AIDicomSlice = "AIDicomSlice";
        /// <summary>
        /// ID
        /// </summary>
        public string ID { get; set; }

        /// <summary>
        /// 智能切割配置Json字符串
        /// </summary>
        public string AIDicomSlice { get; set; }
    }
}
