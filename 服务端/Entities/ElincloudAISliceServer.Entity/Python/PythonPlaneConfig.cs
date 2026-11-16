using Newtonsoft.Json;

namespace ElincloudAISliceServer.Entity
{
    /// <summary>
    /// Ptyhon用的切割配置
    /// </summary>
    public class PythonPlaneConfig
    {
        /// <summary>
        /// 重建状位：横状位、矢状位或冠状位。<
        /// /summary>
        [JsonProperty("Plane")]
        public string Plane { get; set; } = string.Empty;

        /// <summary>
        /// 切割方式：MPR、CPR 或椎间隙。
        /// </summary>
        [JsonProperty("SliceType")]
        public string SliceType { get; set; } = string.Empty;

        private double sliceThickness = 0;
        /// <summary>
        /// 层厚
        /// </summary>
        [JsonProperty("SliceThickness")]
        public double SliceThickness
        {
            get { return sliceThickness; }
            set { sliceThickness = value; }
        }

        /// <summary>
        /// 请求层间距，单位 mm。
        /// </summary>
        [JsonProperty("SliceSpacing")]
        public double SliceSpacing { get; set; }

        /// <summary>
        /// 请求生成的切片数量。
        /// </summary>
        [JsonProperty("SliceImageCount")]
        public int SliceImageCount { get; set; }

        /// <summary>
        /// 填充比例（百分比制）
        /// 0为不缩放
        /// </summary>
        [JsonProperty("FillPercent")]
        public int FillPercent { get; set; }
        
        /// <summary>
        /// 同一切割条件需要生成的窗位集合。
        /// </summary>
        [JsonProperty("WindowPresets")]
        public List<string> WindowPresets { get; set; } = new List<string>();
    }
}
