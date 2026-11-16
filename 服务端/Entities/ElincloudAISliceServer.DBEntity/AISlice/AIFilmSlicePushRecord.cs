namespace ElincloudAISliceServer.DBEntity
{
    /// <summary>
    /// AI 胶片切割推送记录
    /// </summary>
    public class AIFilmSlicePushRecord
    {
        /// <summary>
        /// 主键ID
        /// </summary>
        public string ID { get; set; } = string.Empty;

        /// <summary>
        /// 切割记录结果ID
        /// </summary>
        public string AIFilmSliceResultID { get; set; } = string.Empty;

        /// <summary>
        /// 影像结果路径
        /// </summary>
        public string ImagePath { get; set; } = string.Empty;

        /// <summary>
        /// 推送状态
        /// </summary>
        public int PushStatus { get; set; } = 0;

        /// <summary>
        /// 推送日志
        /// </summary>
        public string PushMessage { get; set; } = string.Empty;

        /// <summary>
        /// 推送日期
        /// </summary>
        public DateTime PushDate { get; set; } = new DateTime(1900, 1, 1);

        /// <summary>
        /// 接收方IP
        /// </summary>
        public string IP { get; set; } = string.Empty;

        /// <summary>
        /// 接收方AE
        /// </summary>
        public string AE { get; set; } = string.Empty;

        /// <summary>
        /// 接收方端口号
        /// </summary>
        public int Port { get; set; } = 0;
    }
}