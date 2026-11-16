namespace ElincloudServer.Common
{
    internal class MatchPoint
    {
        public int Row { get; set; }
        public int Column { get; set; }
        public string Content { get; set; }
        public bool IsMatched { get; set; }
        public int Weight { get; set; }
    }
}
