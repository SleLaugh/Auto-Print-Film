using ElincloudServer.Entity;

namespace ElincloudAISliceServer.Entity
{
    public class DicomMachineConfigInfo
    {
        public List<DicomMachineInfo> DicomMachines { get; set; } = new List<DicomMachineInfo>();

        public HospitalInfo Hospital { get; set; } = new HospitalInfo();
    }
}
