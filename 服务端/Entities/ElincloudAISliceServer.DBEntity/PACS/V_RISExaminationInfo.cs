namespace ElincloudAISliceServer.DBEntity
{
    public class V_RISExaminationInfo : ElincloudServer.DBEntity.V_RISExaminationInfo
    {
        /// <summary>
        /// 切割影像状态 
        /// </summary>
        public int SliceImageStatus { get; set; } = 0;

        /// <summary>
        /// 打印次数
        /// </summary>
        public int PrintNumber { get; set; } = 0;

        /// <summary>
        /// 设备 
        /// </summary>
        public string MachineKey { get; set; } = string.Empty;
    }
}
