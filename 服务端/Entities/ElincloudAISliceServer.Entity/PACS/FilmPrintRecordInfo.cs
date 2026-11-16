namespace ElincloudAISliceServer.Entity
{
    public class FilmPrintRecordInfo : ElincloudServer.Entity.FilmPrintRecordInfo
    {
        /// <summary>
        /// SeriesInstanceUID
        /// </summary>
        public string SeriesInstanceUID { get; set; } = string.Empty;

        /// <summary>
        /// 检查部位
        /// </summary>
        public string StudyPartName { get; set; } = string.Empty;

        /// <summary>
        /// 模板ID
        /// </summary>
        public string LayoutTemplateID { get; set; } = string.Empty;
    }
}
