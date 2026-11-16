namespace ElincloudAISliceServer.DBEntity
{
    /// <summary>
    /// 切割影像预设配置
    /// </summary>
    public class AIDicomSlicePresetConfig
    {
        public static string _AIDicomSlicePreset = "AIDicomSlicePreset";
        /// <summary>
        /// hospitalCode
        /// </summary>
        public string HospitalCode { get; set; }

        /// <summary>
        /// 配置Json字符串
        /// </summary>
        public string AIDicomSlicePreset { get; set; }
    }
}
