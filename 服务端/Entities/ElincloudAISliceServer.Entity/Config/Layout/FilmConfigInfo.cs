namespace ElincloudAISliceServer.Entity
{
    /// <summary>
    /// 胶片配置
    /// </summary>
    public class FilmConfigInfo
    {
        /// <summary>
        /// 当前胶片下标
        /// </summary>
        public int FilmIndex { get; set; }

        /// <summary>
        /// 行数
        /// </summary>
        public int RowNum { get; set; }

        /// <summary>
        /// 列量
        /// </summary>
        public int ColumnNum { get; set; }

        /// <summary>
        /// 排版配置
        /// </summary>
        public List<LayoutCellInfo> LayoutCellInfos { get; set; } = new List<LayoutCellInfo>();
    }
}
