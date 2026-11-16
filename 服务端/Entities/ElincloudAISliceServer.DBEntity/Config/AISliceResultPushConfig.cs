namespace ElincloudAISliceServer.DBEntity
{
    /// <summary>
    /// AI切割结果推送配置
    /// </summary>
    public class AISliceResultPushConfig
    {
        public static string _AISliceResultPush = "AISliceResultPush";
        /// <summary>
        /// hospitalCode
        /// </summary>
        public string HospitalCode { get; set; }

        /// <summary>
        /// 配置Json字符串
        /// </summary>
        public string AISliceResultPush { get; set; }
    }
}
