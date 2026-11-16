using Microsoft.AspNetCore.Mvc;

namespace ElincloudServer.Controllers
{
    public class ArchiveBLServiceController : BaseController
    {
        public IActionResult Index()
        {
            return View();
        }
    }
}
