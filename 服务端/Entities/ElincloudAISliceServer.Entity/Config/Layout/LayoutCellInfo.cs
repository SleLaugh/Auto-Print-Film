using ElincloudAISliceServer.Entity;

namespace ElincloudAISliceServer.Entity
{
    public class LayoutCellInfo
    {
        /// <summary>
        /// 胶片下标
        /// </summary>
        public int FilmIndex { get; set; }

        /// <summary>
        /// 所在行下标
        /// </summary>
        public int RowIndex { get; set; }

        /// <summary>
        /// 所在列下标
        /// </summary>
        public int ColumnIndex { get; set; }

        /// <summary>
        /// 切割方案 ID
        /// </summary>
        public string SlicePlanID { get; set; } = string.Empty;

        /// <summary>
        /// 切割区域
        /// </summary>
        public SliceRegions Region { get; set; } = SliceRegions.Default;

        /// <summary>
        /// 切割方案输出影像的下标
        /// </summary>
        public int ImageIndex { get; set; }
    }
}
