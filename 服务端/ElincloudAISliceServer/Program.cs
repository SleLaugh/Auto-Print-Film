using CoreWCF;
using CoreWCF.Channels;
using CoreWCF.Configuration;
using CoreWCF.Description;
using ElincloudServer.BLServices;
using ElincloudServer.Common;
using ElincloudServer.CommonBLService;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.AspNetCore.Server.Kestrel.Core;
using Microsoft.AspNetCore.Server.Kestrel.Https;
using Microsoft.AspNetCore.SignalR;
using Microsoft.Extensions.FileProviders;
using System.IO;
using System.Net;
using System.Security.Cryptography.X509Certificates;
using System.Text;

namespace ElincloudServer
{
    public class Program
    {
        public static void Main(string[] args)
        {
            // 注册编码提供程序
            Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);

            var builder = WebApplication.CreateBuilder(args);

            // 注册跨域策略
            builder.Services.AddCors(options =>
            {
                var allowedOrigins = builder.Configuration.GetSection("Cors:AllowedOrigins").Get<string[]>();
                options.AddPolicy("AllowedOrigins", policy =>
                {
                    policy
                          .WithOrigins(allowedOrigins ?? new string[] { "http://localhost:5000" })
                          .AllowAnyHeader()
                          .AllowAnyMethod()
                          .AllowCredentials();  // 如果用Cookie、认证要加这个
                });
            });

            builder.Services.AddControllersWithViews(options =>
            {
                // 全局过滤器
                options.Filters.Add(new AuthorizeFilter());  // 注册全局权限验证过虑器
                options.Filters.Add(new ResourcesFilter()); // 注册全局资源过虑器
                options.Filters.Add(new ActionSecurityFilter()); // 注册全局接口安全过虑器
                options.Filters.Add(new LogAuditFilter());  //注册日志审计过虑器
            });

            builder.Services.AddControllers();

            builder.Services.AddHttpContextAccessor();

            //
            //JSON默认转换时不进行大小写转换 
            //builder.Services.AddControllers().AddJsonOptions(options =>
            //{
            //    options.JsonSerializerOptions.PropertyNamingPolicy = null; // 关闭属性名转换策略
            //});
            builder.Services.AddControllers().AddNewtonsoftJson(options =>
            {
                options.SerializerSettings.ContractResolver = new Newtonsoft.Json.Serialization.DefaultContractResolver();
            });
            //限制文件上传大小
            builder.Services.Configure<FormOptions>(options =>
            {
                options.MultipartBodyLengthLimit = 314572800; // 300 MB
            });
            // WCF服务配置
            builder.Services.AddServiceModelServices();
            builder.Services.AddServiceModelMetadata();
            builder.Services.AddSingleton<IServiceBehavior, UseRequestHeadersForMetadataAddressBehavior>();
            // 允许同步IO，传输流
            builder.Services.Configure<KestrelServerOptions>(options =>
            {
                options.AllowSynchronousIO = true;
            });

            // 添加 SignalR
            builder.Services.AddSignalR().AddJsonProtocol(options =>
            {
                options.PayloadSerializerOptions.PropertyNamingPolicy = null; // 取消小驼峰转换
            });

            builder.Services.Configure<IISServerOptions>(options =>
            {
                options.AllowSynchronousIO = true;
            });
            //读取文件中Kestrel的配置（启动端口）
            builder.Host.ConfigureAppConfiguration((context, config) =>
            {
                var basePath = AppContext.BaseDirectory; // 获取程序根目录
                config.SetBasePath(basePath);
                config.AddJsonFile("appsettings.json", optional: false, reloadOnChange: true);
            });

            var app = builder.Build();
            app.UseCors("AllowedOrigins");
            app.UseAuthorization();

            //初始化自定义扩展服务
            app.InitExtensions();

            //配置默认文件，使得访问根路径时返回 index.html
            if (app.Environment.IsProduction() && Common.Utils.ConfigUtil.Deployment == "1")
            {
                //配置静态文件服务
                app.UseDefaultFiles(new DefaultFilesOptions
                {
                    FileProvider = new PhysicalFileProvider(ElincloudServer.Common.Utils.StringUtil.Combine(AppContext.BaseDirectory, "wwwroot", "WebPACSClient")),
                    RequestPath = ""  // 不需要设置为 index.html，设置为空字符串表示根路径
                });
                app.UseStaticFiles(new StaticFileOptions
                {
                    FileProvider = new PhysicalFileProvider(ElincloudServer.Common.Utils.StringUtil.Combine(AppContext.BaseDirectory, "wwwroot", "WebPACSClient")),
                    RequestPath = "",  // 设置为空，意味着从根路径访问静态文件
                });

                //配置静态文件服务
                app.UseDefaultFiles(new DefaultFilesOptions
                {
                    FileProvider = new PhysicalFileProvider(ElincloudServer.Common.Utils.StringUtil.Combine(AppContext.BaseDirectory, "wwwroot", "WebPACSDicomViewer")),
                    RequestPath = "/WebPACSDicomViewer"  // 不需要设置为 index.html，设置为空字符串表示根路径
                });
                app.UseStaticFiles(new StaticFileOptions
                {
                    FileProvider = new PhysicalFileProvider(ElincloudServer.Common.Utils.StringUtil.Combine(AppContext.BaseDirectory, "wwwroot", "WebPACSDicomViewer")),
                    RequestPath = "/WebPACSDicomViewer",  // 设置为空，意味着从根路径访问静态文件
                });
            }

            app.MapControllers();

            app.Run();
        }
    }
}
