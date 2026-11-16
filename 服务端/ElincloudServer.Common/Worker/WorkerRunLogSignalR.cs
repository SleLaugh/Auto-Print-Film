using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.SignalR;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using System.Net.Sockets;
using System.Net;

namespace ElincloudServer.Common
{
    public static class WorkerRunLogSignalR
    {
        /// <summary>
        /// 扩展方法，添加默认的 WorkerLogHub SignalR 服务和路由
        /// </summary>
        /// <param name="hostBuilder">主机构建器</param>
        /// <param name="port">监听端口</param>
        /// <returns>配置后的主机构建器</returns>
        public static IHostBuilder AddSignalRHost(this IHostBuilder hostBuilder, int port)
        {
            // 调用泛型版本，默认使用 WorkerLogHub
            return AddSignalRHost<WorkerRunLogHub>(hostBuilder, port);
        }

        /// <summary>
        /// 泛型扩展方法，添加指定 Hub 类型的 SignalR 服务和路由
        /// </summary>
        /// <typeparam name="THub">SignalR Hub 类型</typeparam>
        /// <param name="hostBuilder">主机构建器</param>
        /// <param name="hubPath">Hub 路由路径</param>
        /// <param name="port">监听端口</param>
        /// <returns>配置后的主机构建器</returns>
        public static IHostBuilder AddSignalRHost<THub>(this IHostBuilder hostBuilder, int port)
            where THub : Hub
        {
            // 检查端口是否可用
            if (!IsPortFree(port))
            {
                // 端口被占用：静默跳过，不注册任何东西
                return hostBuilder;
            }

            // 端口空闲：正常注册 SignalR 和 WebHost
            return hostBuilder
                .ConfigureServices((context, services) =>
                {
                    var config = new ConfigurationBuilder().AddJsonFile("appsettings.json").Build();
                    var allowedOrigins = config.GetSection("Cors:AllowedOrigins").Get<string[]>();

                    services.AddCors(options =>
                    {
                        options.AddPolicy("AllowedOrigins", policy =>
                        {
                            if (allowedOrigins == null || allowedOrigins.Length == 0)
                            {
                                // 注意：AllowAnyOrigin() + AllowCredentials() 是非法组合，浏览器会拒绝

                                // 所以如果需要 credentials（如 cookies、Authorization header），必须指定具体 origins

                                policy.AllowAnyOrigin()
                                      .AllowAnyHeader()
                                      .AllowAnyMethod();
                                // 注意：这里去掉了 .AllowCredentials()
                            }
                            else
                            {
                                policy.WithOrigins(allowedOrigins)
                                      .AllowAnyHeader()
                                      .AllowAnyMethod()
                                      .AllowCredentials(); // 
                            }
                        });
                    });

                    services.AddSignalR();

                    if (typeof(THub) == typeof(WorkerRunLogHub))
                    {
                        services.AddHostedService<HubContextInitHostedService>();
                    }
                })
                .ConfigureWebHostDefaults(webBuilder =>
                {
                    webBuilder.ConfigureKestrel(serverOptions => serverOptions.ListenAnyIP(port));
                    webBuilder.Configure(app =>
                    {
                        app.UseRouting();
                        app.UseCors("AllowedOrigins");
                        app.UseEndpoints(endpoints =>
                        {
                            endpoints.MapHub<THub>("WorkerRunLogHub");
                        });
                    });
                });
        }

        private static bool IsPortFree(int port)
        {
            try
            {
                var listener = new TcpListener(IPAddress.Loopback, port);
                listener.Start(); // 尝试绑定
                listener.Stop();  // 立即释放
                return true;
            }
            catch (SocketException)
            {
                return false; // 端口已被占用
            }
        }

        /// <summary>
        /// 用于在主机启动时初始化 WorkerLogHub 的 HubContext，并注入到 WorkerConfigUtilBase
        /// </summary>
        private sealed class HubContextInitHostedService : IHostedService
        {
            private readonly IServiceProvider _sp;

            public HubContextInitHostedService(IServiceProvider sp)
            {
                _sp = sp;
            }

            /// <summary>
            /// 启动时获取 HubContext 并设置到静态类
            /// </summary>
            public Task StartAsync(CancellationToken cancellationToken)
            {
                var hubContext = _sp.GetRequiredService<IHubContext<WorkerRunLogHub>>();
                WorkerConfigUtilBase.SetHubContext(hubContext);
                return Task.CompletedTask;
            }

            public Task StopAsync(CancellationToken cancellationToken) => Task.CompletedTask;
        }
    }
}