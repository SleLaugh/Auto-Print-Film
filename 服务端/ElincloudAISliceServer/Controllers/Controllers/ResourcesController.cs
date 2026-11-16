using Azure;
using ElincloudServer.Common;
using ElincloudServer.Entity;
using Microsoft.AspNetCore.Mvc;

namespace ElincloudServer.Controllers
{
    [SkipAuthorizeFilterAttribute]
    [SkipActionSecurityFilterAttribute]
    [Route("Resources")]
    public class ResourcesController : Controller
    {
        private readonly IConfiguration _configuration;
        public ResourcesController(IConfiguration configuration)
        {
            _configuration = configuration;
        }

        [HttpGet("Vist")]
        public IActionResult Vist(string k, string op, string d, string ab)
        {
            string errorURL = "ErrorView";
            try
            {
                string encryptKey = ElincloudServer.Common.Utils.StringUtil.Decrypt(ApplicationEx.EncryptKey);
                string dateStr = ElincloudServer.Common.Utils.StringUtil.Decrypt(d, encryptKey);
                bool isAbsolute = false;
                if (!string.IsNullOrEmpty(ab))
                {
                    isAbsolute = Convert.ToBoolean(ElincloudServer.Common.Utils.StringUtil.Decrypt(ab, encryptKey));
                }

                DateTime date = new DateTime(1900, 1, 1);
                DateTime.TryParse(dateStr, out date);
                if (!DateTime.TryParse(dateStr, out date))
                {
                    return RedirectToAction(errorURL, new { errorMessage = "无效的时间格式！" });
                }
                else if ((DateTime.Now - date).TotalMinutes >= 480)
                {
                    return RedirectToAction(errorURL, new { errorMessage = "拒绝访问,请求链接已失效！" });
                }
                else
                {
                    string relaFilePath = ElincloudServer.Common.Utils.StringUtil.Decrypt(k, encryptKey);
                    string fullFilePath = string.Empty;
                    if (isAbsolute)
                    {
                        fullFilePath = relaFilePath;
                    }
                    else
                    {
                        fullFilePath = Common.Utils.StringUtil.Combine(Common.Utils.ConfigUtil.FileStoragePath, relaFilePath);
                    }
                    if (System.IO.File.Exists(fullFilePath))
                    {
                        string extension = Path.GetExtension(fullFilePath);
                        string opAction = ElincloudServer.Common.Utils.StringUtil.Decrypt(op, encryptKey);
                        //通过配置文件获取 MIME 类型映射，确保浏览器正确显示文件，而不是默认下载。
                        var contentTypeDic = _configuration.GetSection("ContentTypeDic").Get<Dictionary<string, string>>();
                        string contentType = string.Empty;
                        contentTypeDic.TryGetValue(extension.ToLower(), out contentType);
                        //如果没有获取到文件的类型，通过文件流信息获取真实的文件类型
                        if (string.IsNullOrEmpty(contentType))
                        {
                            contentType = GetRealFileType(fullFilePath);
                        }
                        if (string.IsNullOrEmpty(contentType))
                        {
                            return RedirectToAction(errorURL, new { errorMessage = "未知的文件类型！" });
                        }
                        else
                        {
                            string requestOrigin = Request.Headers["Origin"].ToString();
                            Response.Headers.Append("Access-Control-Allow-Origin", requestOrigin);
                            if (opAction == URLOperations.Download.ToString())
                            {
                                return PhysicalFile(fullFilePath, $"{contentType}", Utils.StringUtil.GetFileName(fullFilePath));
                            }
                            else if (opAction == URLOperations.Browse.ToString())
                            {
                                var fileName = Utils.StringUtil.GetFileName(fullFilePath);

                                // RFC 5987 格式，解决中文/特殊字符
                                var encodedFileName = Uri.EscapeDataString(fileName);

                                Response.Headers.Add("Content-Disposition", $"inline; filename*=UTF-8''{encodedFileName}");
                                return PhysicalFile(fullFilePath, $"{contentType}");
                            }
                            else
                            {
                                return RedirectToAction(errorURL, new { errorMessage = "未知的操作类型！" });
                            }
                        }
                    }
                    else
                    {
                        return RedirectToAction(errorURL, new { errorMessage = "要访问的资源已被清理,请联系管理员！" });
                    }
                }
            }
            catch (Exception ex)
            {
                return RedirectToAction(errorURL, new { errorMessage = ex.Message });
            }
        }

        private string GetRealFileType(string filePath)
        {
            string ret = string.Empty;
            byte[] buffer = new byte[8];
            using (FileStream fs = new FileStream(filePath, FileMode.Open, FileAccess.Read))
            {
                fs.Read(buffer, 0, buffer.Length);
            }
            string hex = BitConverter.ToString(buffer);
            if (hex.StartsWith("25-50-44-46"))
            {
                ret = "application/pdf";
            }
            if (hex.StartsWith("FF-D8-FF"))
            {
                ret = "image/jpeg";
            }
            if (hex.StartsWith("89-50-4E-47"))
            {
                ret = "image/jpeg";
            }
            return ret;
        }

        public IActionResult ErrorView(string errorMessage)
        {
            ViewData["Msg"] = errorMessage;
            return View();
        }
    }
}
