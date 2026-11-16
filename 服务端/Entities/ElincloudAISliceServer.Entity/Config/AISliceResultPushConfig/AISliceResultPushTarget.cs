namespace ElincloudAISliceServer.Entity
{
    public class AISliceResultPushTarget
    {
        /// <summary>
        /// 名称
        /// </summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>
        /// 是否启用
        /// </summary>
        public bool IsEnabel { get; set; } = true;

        /// <summary>
        /// ip
        /// </summary>
        public string IP { get; set; } = string.Empty;

        /// <summary>
        /// 端口
        /// </summary>
        public int Port { get; set; } = 0;

        /// <summary>
        /// AE
        /// </summary>
        public string AE { get; set; } = string.Empty;

        /// <summary>
        /// 备注
        /// </summary>
        public string Remark { get; set; } = string.Empty;

        /// <summary>
        /// 切割方案ID
        /// </summary>
        public List<string> SlicePlanIDs { get; set; } = new List<string>();
    }
}
