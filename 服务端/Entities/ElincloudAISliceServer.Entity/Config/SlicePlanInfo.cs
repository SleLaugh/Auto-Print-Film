namespace ElincloudAISliceServer.Entity
{
    /// <summary>
    /// 切割方案
    /// </summary>
    public class SlicePlanConfigInfo
    {
        /// <summary>
        /// 默认构造函数
        /// </summary>
        public SlicePlanConfigInfo()
        {
        }

        private string id = string.Empty;
        /// <summary>
        /// 主键ID
        /// </summary>
        public string ID
        {
            get { return id; }
            set { id = value; }
        }

        private string name = string.Empty;
        /// <summary>
        /// 切割方案名称
        /// </summary>
        public string Name
        {
            get { return name; }
            set { name = value; }
        }

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



        private string slicePartName = string.Empty;
        /// <summary>
        /// 切割部位
        /// </summary>
        public string SlicePartName
        {
            get { return slicePartName; }
            set { slicePartName = value; }
        }

        private PlaneEnums plane = PlaneEnums.横状位;
        /// <summary>
        /// 状位
        /// </summary>
        public PlaneEnums Plane
        {
            get { return plane; }
            set { plane = value; }
        }

        private SliceTypes sliceType = SliceTypes.MPR;
        /// <summary>
        /// 切割方式
        /// </summary>
        public SliceTypes SliceType
        {
            get { return sliceType; }
            set { sliceType = value; }
        }

        private double sliceThickness = 0;
        /// <summary>
        /// 层厚
        /// </summary>
        public double SliceThickness
        {
            get { return sliceThickness; }
            set { sliceThickness = value; }
        }

        private double sliceSpacing = 0;
        /// <summary>
        /// 层间距(mm)
        /// </summary>
        public double SliceSpacing
        {
            get { return sliceSpacing; }
            set { sliceSpacing = value; }
        }

        private int sliceImageCount = 0;
        /// <summary>
        /// 切割影像数量
        /// </summary>
        public int SliceImageCount
        {
            get { return sliceImageCount; }
            set { sliceImageCount = value; }
        }

        private string windowPreset = string.Empty;
        /// <summary>
        /// 窗宽窗位
        /// </summary>
        public string WindowPreset
        {
            get { return windowPreset; }
            set { windowPreset = value; }
        }

        /// <summary>
        /// 填充比例（百分比制）
        /// 0为不缩放
        /// </summary>
        public int FillPercent { get; set; } = 0;

        /// <summary>
        /// 切割区域
        /// </summary>
        public List<SliceRegions> Regions { get; set; } = new List<SliceRegions>();
    }
}
