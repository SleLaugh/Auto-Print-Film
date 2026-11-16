using Microsoft.AspNetCore.Mvc.ModelBinding;
using Microsoft.AspNetCore.SignalR;

namespace ElincloudServer.Common
{
    public abstract class WorkerConfigUtilBase
    {
        /// <summary>
        /// 保存 WorkerRunLogHub 的 SignalR HubContext 实例，用于推送消息
        /// </summary>
        private static IHubContext<WorkerRunLogHub>? _hubContext;

        /// <summary>
        /// 设置 HubContext，通常由依赖注入容器初始化后调用，确保全局唯一实例
        /// </summary>
        /// <param name="hubContext">SignalR HubContext 实例</param>
        public static void SetHubContext(IHubContext<WorkerRunLogHub> hubContext)
        {
            _hubContext = hubContext;
        }

        #region 线程池

        private readonly object threadLocker = new object();

        public void PushActionToThreadPool(WaitCallback callback, object para = null)
        {
            lock (threadLocker)
            {
                if (para != null)
                {
                    ThreadPool.QueueUserWorkItem(callback, para);
                }
                else
                {
                    ThreadPool.QueueUserWorkItem(callback);
                }
                //这里的sleep为防止外部代码在一个for循环将资源全部占用的问题
                Thread.Sleep(1);
            }
        }

        public void InitThreadPool(int minThreads, int maxThreads)
        {
            if (minThreads <= 0)
            {
                throw new Exception("最小线程数量不能小于0!");
            }
            if (maxThreads <= 0)
            {
                throw new Exception("最大线程数量不能小于0!");
            }
            if (maxThreads < minThreads)
            {
                throw new Exception("最小线程数量不能大于最大线程数量!");
            }
            ThreadPool.SetMinThreads(minThreads, minThreads);
            ThreadPool.SetMaxThreads(maxThreads, maxThreads);
        }

        #endregion

        public string WorkerName
        {
            get; set;
        }

        public string GetDicomMachineKey(string ip, string clientAE)
        {
            return string.Format("{0}({1})", clientAE, ip);
        }

        #region 日志相关

        /// <summary>
        /// 
        /// </summary>
        /// <param name="msg">消息内容</param>
        /// <param name="isWriteLogToFile">是否记录到日志文件</param>
        /// <param name="childServerName">子服务名称</param>
        /// <param name="ex">异常信息</param>
        public void PrintLogAction(string msg, bool isWriteLogToFile = false, string childServerName = "", Exception ex = null)
        {
            // 1. 添加时间戳
            string timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
            string logMessage = $"[{timestamp}] {msg}";
            // 2. 写入日志文件
            if (isWriteLogToFile)
            {
                if (ex != null)
                {
                    ElincloudServer.Common.Utils.LogUtil.RecordErrorLog(logMessage, ex);
                }
                else
                {
                    ElincloudServer.Common.Utils.LogUtil.RecordRunLog(logMessage);
                }
            }
            // 3. 添加服务器标识
            childServerName = string.IsNullOrWhiteSpace(childServerName) ? WorkerName : childServerName;
            logMessage = $"{childServerName}@{DateTime.UtcNow.Ticks.ToString()}@{logMessage}";
            // 4. 通过SignalR发送到客户端
            SendToClients(logMessage);
        }

        public void PrintStatusLogAction(string msg, string childServerName = "", bool isDisplayTime = true)
        {
            string timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
            string logMessage = isDisplayTime ? $"[{timestamp}] {msg}" : msg;
            childServerName = string.IsNullOrWhiteSpace(childServerName) ? WorkerName : childServerName;
            logMessage = $"{childServerName} @{logMessage}";
            WriteWorkerRunStatusLog(logMessage);
        }
        private static readonly object _logLock = new object();
        private void WriteWorkerRunStatusLog(string logMessage)
        {
            lock (_logLock)
            {
                try
                {
                    string logDir = Common.Utils.StringUtil.Combine(AppDomain.CurrentDomain.BaseDirectory, ApplicationEx.WorkerServerRunStatuslogName);
                    if (!Directory.Exists(logDir))
                    {
                        Common.Utils.FileUtil.CreateDirectory(logDir);
                    }
                    string fileFullName = Common.Utils.StringUtil.Combine(logDir, $"{WorkerName}.txt");
                    File.AppendAllText(fileFullName, logMessage + Environment.NewLine);
                }
                catch (Exception ex)
                {
                    // worker服务运行状态写入失败时记录到日志
                    ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("写入worker服务运行状态日志异常！", ex);
                }
            }
        }

        public void DeleteWorkerRunStatusLogContent()
        {
            try
            {
                string fileFullName = Common.Utils.StringUtil.Combine(AppDomain.CurrentDomain.BaseDirectory, ApplicationEx.WorkerServerRunStatuslogName, $"{WorkerName}.txt");
                if (File.Exists(fileFullName))
                {
                    File.Delete(fileFullName);
                }
            }
            catch (Exception ex)
            {
                // worker服务运行状态写入失败时记录到日志
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("删除worker服务运行状态日志异常！", ex);
            }
        }

        private void SendToClients(string logMessage)
        {
            try
            {
                if (_hubContext != null)
                {
                    // 使用Task.Run避免阻塞当前线程
                    Task.Run(() => _hubContext.Clients.All.SendAsync("WorkerRunLogCallBack", logMessage));
                }
            }
            catch (Exception ex)
            {
                // SignalR发送失败时记录到日志
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("通过SignalR发送消息到客户端失败！", ex);
            }
        }

        public void PrintDebugLogAction(string msg)
        {
            msg = "[" + DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss") + "]" + msg;
            ElincloudServer.Common.Utils.LogUtil.RecordRunLog(msg);
        }

        #endregion

        public bool InitSuccess { get; protected set; }

        /// <summary>
        /// 服务运行的参数信息
        /// </summary>
        public string ServerRunParmInfo = string.Empty;
    }
}
