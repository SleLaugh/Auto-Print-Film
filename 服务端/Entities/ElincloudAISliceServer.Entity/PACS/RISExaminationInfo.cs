namespace ElincloudAISliceServer.Entity
{
    public class RISExaminationInfo : ElincloudServer.Entity.RISExaminationInfo
    {
        /// <summary>
        /// 切割影像状态 
        /// </summary>
        public SliceImageStatus SliceImageStatus { get; set; } = SliceImageStatus.正在切割;

        /// <summary>
        /// 打印次数
        /// </summary>
        public int PrintNumber { get; set; } = 0;

        /// <summary>
        /// 是否加急
        /// </summary>
        public bool IsUrgent { get; set; } = false;

        /// <summary>
        /// 设备 
        /// </summary>
        public string MachineKey { get; set; } = string.Empty;

        /// <summary>
        /// 设备名称
        /// </summary>
        public string MachineName { get; set; } = string.Empty;
    }
}
