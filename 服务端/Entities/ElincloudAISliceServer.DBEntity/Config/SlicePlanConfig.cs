using ElincloudServer.DBEntity;

namespace ElincloudAISliceServer.DBEntity
{
    /// <summary>
    /// 切割方案
    /// </summary>
    public class SlicePlanConfig : BaseDBEntity
    {
        public static string _Name = "Name";
        public static string _IsEnabel = "IsEnabel";
        public static string _StudyTypeName = "StudyTypeName";
        public static string _StudyPartName = "StudyPartName";
        public static string _MinScanThickness = "MinScanThickness";
        public static string _MaxScanThickness = "MaxScanThickness";

        public static string _SlicePartName = "SlicePartName";
        public static string _Plane = "Plane";
        public static string _SliceType = "SliceType";
        public static string _SliceThickness = "SliceThickness";
        public static string _SliceSpacing = "SliceSpacing";
        public static string _SliceImageCount = "SliceImageCount";
        public static string _FillPercent = "FillPercent";
        public static string _WindowPreset = "WindowPreset";

        /// <summary>
        /// 默认构造函数
        /// </summary>
        public SlicePlanConfig()
        {
        }

        /// <summary>
        /// 主键ID
        /// </summary>
        public string ID { get; set; } = string.Empty;

        /// <summary>
        /// 切割方案名称
        /// </summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>
        /// 是否启用
        /// </summary>
        public bool IsEnabel { get; set; } = true;

        /// <summary>
        /// 检查类型
        /// </summary>
        public string StudyTypeName { get; set; }

        /// <summary>
        /// 检查部位
        /// </summary>
        public string StudyPartName { get; set; }

        /// <summary>
        /// 最小层厚
        /// </summary>
        public double MinScanThickness { get; set; }

        /// <summary>
        /// 最大层厚
        /// </summary>
        public double MaxScanThickness { get; set; }



        /// <summary>
        /// 切割部位
        /// </summary>
        public string SlicePartName { get; set; } = string.Empty;

        /// <summary>
        /// 状位
        /// </summary>
        public int Plane { get; set; } = 0;

        /// <summary>
        /// 切割方式
        /// </summary>
        public int SliceType { get; set; } = 0;

        /// <summary>
        /// 层厚
        /// </summary>
        public double SliceThickness { get; set; } = 0;

        /// <summary>
        /// 层间距(mm)
        /// </summary>
        public double SliceSpacing { get; set; } = 0;

        /// <summary>
        /// 切割影像数量
        /// </summary>
        public int SliceImageCount { get; set; } = 0;

        /// <summary>
        /// 填充比例（百分比制）
        /// 0为不缩放
        /// </summary>
        public int FillPercent { get; set; } = 0;

        /// <summary>
        /// 窗宽窗位
        /// </summary>
        public string WindowPreset { get; set; } = string.Empty;
    }
}
