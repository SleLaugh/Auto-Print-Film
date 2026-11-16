namespace ElincloudAISliceServer.Entity
{
    /// <summary>
    /// 窗宽窗位
    /// </summary>
    public class WindowPreset
    {
        public WindowPreset()
        {

        }
        public WindowPreset(string name, int windowWidth, int windowLevel)
        {
            Name = name;
            WindowWidth = windowWidth;
            WindowLevel = windowLevel;
        }


        /// <summary>
        /// 名称
        /// </summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>
        /// 窗宽
        /// </summary>
        public int WindowWidth { get; set; }

        /// <summary>
        /// 窗位
        /// </summary>
        public int WindowLevel { get; set; }
    }
}
