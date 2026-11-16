using Microsoft.AspNetCore.Mvc;

namespace ElincloudServer
{
    public class ResoucecCache
    {
        public PathString Method { get; set; }
        public DateTime CacheDate { get; set; }
        public IActionResult Result { get; set; }
    }
}
