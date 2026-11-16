namespace ElincloudAISliceServer.Entity
{
    /// <summary>
    /// AI切割结果推送配置
    /// </summary>
    public class AISliceResultPushConfigInfo
    {
        public string HospitalCode { get; set; }

        /// <summary>
        /// 推送设备配置
        /// </summary>
        public List<AISliceResultPushTarget> AISliceResultPushTarget { get; set; } = new List<AISliceResultPushTarget>();
    }
}
