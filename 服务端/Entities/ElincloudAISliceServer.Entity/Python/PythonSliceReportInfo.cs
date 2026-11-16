using Newtonsoft.Json;

namespace ElincloudAISliceServer.Entity
{
    /// <summary>
    /// Python生成的report.json切割结果
    /// </summary>
    public class PythonSliceReportInfo
    {
        /// <summary>
        /// 本次切割是否执行成功
        /// </summary>
        [JsonProperty("Success")]
        public bool Success { get; set; }

        /// <summary>
        /// 执行失败时返回的错误原因
        /// </summary>
        [JsonProperty("Message")]
        public string Message { get; set; } = string.Empty;

        /// <summary>
        /// 切割部位
        /// </summary>
        [JsonProperty("SlicePart")]
        public string SlicePart { get; set; } = string.Empty;

        /// <summary>
        /// 检查类型
        /// </summary>
        [JsonProperty("StudyType")]
        public string StudyType { get; set; } = string.Empty;

        /// <summary>
        /// 原始扫描层厚
        /// </summary>
        [JsonProperty("ScanThickness")]
        public double? ScanThickness { get; set; }

        /// <summary>
        /// 所有切割配置的返回结果
        /// </summary>
        [JsonProperty("PlaneList")]
        public List<PythonSliceReportPlaneInfo> PlaneList { get; set; } = new List<PythonSliceReportPlaneInfo>();
    }

    /// <summary>
    /// report.json中一项状位切割配置
    /// </summary>
    public class PythonSliceReportPlaneInfo
    {
        /// <summary>
        /// Python为合并后的切割配置生成的标识
        /// </summary>
        [JsonProperty("ConfigId")]
        public string ConfigId { get; set; } = string.Empty;

        /// <summary>
        /// 状位名称，相同状位多项配置时可能带顺序后缀
        /// </summary>
        [JsonProperty("Plane")]
        public string Plane { get; set; } = string.Empty;

        /// <summary>
        /// 切割方式
        /// </summary>
        [JsonProperty("SliceType")]
        public string SliceType { get; set; } = string.Empty;

        /// <summary>
        /// 请求层厚
        /// </summary>
        [JsonProperty("SliceThickness")]
        public double? SliceThickness { get; set; }

        /// <summary>
        /// 请求层间距
        /// </summary>
        [JsonProperty("SliceSpacing")]
        public double? SliceSpacing { get; set; }

        /// <summary>
        /// 请求切片数量
        /// </summary>
        [JsonProperty("SliceImageCount")]
        public int? SliceImageCount { get; set; }

        /// <summary>
        /// 填充比例（百分比制）
        /// 0为不缩放
        /// </summary>
        [JsonProperty("FillPercent")]
        public int? FillPercent { get; set; }

        /// <summary>
        /// 该配置返回的所有窗位
        /// </summary>
        [JsonProperty("WindowPresetList")]
        public List<PythonSliceReportWindowInfo> WindowPresetList { get; set; } = new List<PythonSliceReportWindowInfo>();
    }

    /// <summary>
    /// report.json中一项窗位结果
    /// </summary>
    public class PythonSliceReportWindowInfo
    {
        /// <summary>
        /// 窗位名称
        /// </summary>
        [JsonProperty("Name")]
        public string Name { get; set; } = string.Empty;

        /// <summary>
        /// 该窗位下生成的所有区域
        /// </summary>
        [JsonProperty("RegionList")]
        public List<PythonSliceReportRegionInfo> RegionList { get; set; } = new List<PythonSliceReportRegionInfo>();
    }

    /// <summary>
    /// report.json中一项实际切割区域
    /// </summary>
    public class PythonSliceReportRegionInfo
    {
        /// <summary>
        /// 切割区域
        /// </summary>
        [JsonProperty("Region")]
        public string Region { get; set; } = string.Empty;

        /// <summary>
        /// 实际层厚
        /// </summary>
        [JsonProperty("ActualSliceThickness")]
        public double ActualSliceThickness { get; set; }

        /// <summary>
        /// 实际层间距
        /// </summary>
        [JsonProperty("ActualSliceSpacing")]
        public double ActualSliceSpacing { get; set; }

        /// <summary>
        /// 实际生成数量
        /// </summary>
        [JsonProperty("ActualSliceImageCount")]
        public int ActualSliceImageCount { get; set; }

        /// <summary>
        /// 当前区域的影像目录
        /// </summary>
        [JsonProperty("Path")]
        public string Path { get; set; } = string.Empty;
    }
}
