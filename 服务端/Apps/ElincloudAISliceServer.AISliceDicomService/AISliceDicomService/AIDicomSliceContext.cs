using ElincloudAISliceServer.Entity;
using ElincloudServer.DBService;
using ElincloudServer.Entity;

namespace ElincloudAISliceServer.AISliceDicomService
{
    internal class AIDicomSliceContext
    {
        /// <summary>
        /// 
        /// </summary>
        public HospitalInfo Hospital { get; set; }

        /// <summary>
        /// 
        /// </summary>
        public SqlAccess Instance { get; set; }

        /// <summary>
        /// 
        /// </summary>
        public List<SlicePlanConfigInfo>? SlicePlans { get; set; }

        /// <summary>
        /// 
        /// </summary>
        public AIDicomSlicePresetConfigInfo? Preset { get; set; }
    }
}
