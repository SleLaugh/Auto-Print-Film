using ElincloudServer.Entity;

namespace ElincloudAISliceServer.Entity
{
    public class WorkFilmConfigInfo
    {
        public List<WorkFilmFourCornerInfo> FourCornerTemplate { get; set; } = new List<WorkFilmFourCornerInfo>();

        public WorkFilmSettingsInfo GeneralSettingsInfo { get; set; } = new WorkFilmSettingsInfo();
    }
}
