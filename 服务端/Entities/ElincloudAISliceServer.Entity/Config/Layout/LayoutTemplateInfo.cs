namespace ElincloudAISliceServer.Entity
{
    /// <summary>
    /// 智能排片模版
    /// </summary>
    public class LayoutTemplateInfo
    {
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
        /// 模版名称
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
        /// 胶片数量
        /// </summary>
        public int FilmNum { get; set; }

        /// <summary>
        /// 胶片配置
        /// </summary>
        public List<FilmConfigInfo> FilmConfigInfos { get; set; } = new List<FilmConfigInfo>();

        /// <summary>
        /// 用到的切割方案ID
        /// </summary>
        public List<string> SlicePlans { get; set; }
    }
}
