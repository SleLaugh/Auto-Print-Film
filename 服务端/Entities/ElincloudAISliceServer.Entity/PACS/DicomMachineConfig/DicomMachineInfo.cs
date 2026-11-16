using ElincloudServer.Entity;

namespace ElincloudAISliceServer.Entity
{
    public class DicomMachineInfo
    {
        public string Name { get; set; } = string.Empty;

        public string IP { get; set; } = string.Empty;

        public string AE { get; set; } = string.Empty;

        public string Remark { get; set; } = string.Empty;

        public string MachineKey => IP + ":" + AE;

        public HospitalInfo Hospital { get; set; } = new HospitalInfo();
    }
}
