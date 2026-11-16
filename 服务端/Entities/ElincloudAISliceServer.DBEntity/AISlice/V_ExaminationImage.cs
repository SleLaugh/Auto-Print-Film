namespace ElincloudAISliceServer.DBEntity
{
    public class V_ExaminationImage : ExaminationImage
    {
        public string ImagePatientID { get; set; }

        public string ImageExaminationID { get; set; }

        public bool IsUrgent {  get; set; }

        /// <summary>
        /// 设备 
        /// </summary>
        public string MachineKey { get; set; } = string.Empty;
    }
}
