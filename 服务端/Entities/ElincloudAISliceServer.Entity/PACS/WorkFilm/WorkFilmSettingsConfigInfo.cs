namespace ElincloudAISliceServer.Entity
{
    public class WorkFilmSettingsConfigInfo
    {
        public string LoginName { get; set; } = string.Empty;

        public WorkFilmSettingsInfo Settings { get; set; } = new WorkFilmSettingsInfo();
    }
}
