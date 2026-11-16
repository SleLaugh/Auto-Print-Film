using ElincloudServer.DBEntity;

namespace ElincloudAISliceServer.DBEntity
{
    /// <summary>
    /// 智能模版
    /// </summary>
    public class LayoutTemplate : BaseDBEntity
    {
        public static string _Name = "Name";
        public static string _IsEnabel = "IsEnabel";
        public static string _StudyTypeName = "StudyTypeName";
        public static string _StudyPartName = "StudyPartName";
        public static string _FilmNum = "FilmNum";
        public static string _FilmConfigInfos = "FilmConfigInfos";
        public static string _SlicePlans = "SlicePlans";

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
        public string FilmConfigInfos { get; set; }

        /// <summary>
        /// 用到的切割方案ID
        /// </summary>
        public string SlicePlans { get; set; }
    }
}
