using ElincloudServer.BLServices;
using ElincloudServer.Common;
using ElincloudServer.Entity;

namespace ElincloudServer
{
    public static class AppStartExtensions
    {
        public static void InitExtensions(this IApplicationBuilder app)
        {
            //注入HttpContext
            Service.HttpContextAccessor = app.ApplicationServices.GetService<IHttpContextAccessor>();

            //初始化缓存配置
            ResourcesFilter.Init();

            //初始化默认路由
            app.UseRouting();
            app.UseEndpoints(endpoints =>
            {
                endpoints.MapControllerRoute(
                    name: "default",
                    pattern: "{controller=IM}/{action=Index}/{id?}");
            });
        }

        public static void RegisterRoutes(this IApplicationBuilder app)
        {

        }
    }
}
