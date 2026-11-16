namespace ElincloudAISliceServer.Entity
{
    public class DCMImageInfo : ElincloudServer.Entity.DCMImageInfo
    {
        /// <summary>
        /// 层厚
        /// </summary>
        public string ScanThickness { get; set; } = string.Empty;

        /// <summary>
        /// 影像检查部位Tag
        /// </summary>
        public string BodyPartExamined { get; set; } = string.Empty;

        /// <summary>
        /// 序列描述
        /// </summary>
        public string SeriesDescription { get; set; } = string.Empty;

        /// <summary>
        /// 设备 
        /// </summary>
        public string MachineKey { get; set; } = string.Empty;
    }
}
