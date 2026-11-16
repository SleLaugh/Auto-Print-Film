namespace ElincloudAISliceServer.Entity
{
    public class GeneralInfo : ElincloudServer.Entity.GeneralInfo
    {
        /// <summary>
        /// 显示胶片边框信息
        /// </summary>
        public bool ShowFilmFourCornerInfo { get; set; } = true;

        /// <summary>
        /// 视口四角信息
        /// </summary>
        public bool ShowFourCornerInfo { get; set; } = true;

        /// <summary>
        /// 默认显示标尺
        /// </summary>
        public bool DefaultShowScaleOverlay { get; set; } = false;

        /// <summary>
        /// 自动向前补齐
        /// </summary>
        public bool AutoFillForward { get; set; } = false;

        /// <summary>
        /// 主题
        /// </summary>
        public string ThemeId { get; set; } = string.Empty;

        /// <summary>
        /// 左键
        /// </summary>
        public string MouseLeftTool { get; set; } = string.Empty;

        /// <summary>
        /// 右键
        /// </summary>
        public string MouseRightTool { get; set; } = string.Empty;
    }
}
