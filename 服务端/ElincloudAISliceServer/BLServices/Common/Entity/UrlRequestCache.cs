namespace ElincloudServer.BLServices
{
    public class UrlRequestCache
    {
        public string RequestID { get; set; }
        public int RequestCount { get; set; }
        public DateTime LastRequestDate { get; set; }
    }
}
