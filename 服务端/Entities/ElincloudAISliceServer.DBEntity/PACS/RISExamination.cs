namespace ElincloudAISliceServer.DBEntity
{
    public class RISExamination : ElincloudServer.DBEntity.RISExamination
    {
        public const string _MachineKey = "MachineKey";

        public const string _SliceImageStatus = "SliceImageStatus";

        private int sliceImageStatus = 0;
        /// <summary>
        /// 切割影像状态 
        /// </summary>
        public int SliceImageStatus { get; set; } = 0;

        /// <summary>
        /// 设备 
        /// </summary>
        public string MachineKey { get; set; } = string.Empty;
    }
}

