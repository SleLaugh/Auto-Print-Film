
using ElincloudServer.Common.CommonUtil;
using System.Diagnostics;
using System.Text;

namespace ElincloudServer.Common
{
    public class LogUtil
    {
        private readonly Object lockObje = new Object();

        public string baseLogPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, ApplicationEx.ServiceLogFolder);

        public void RecordErrorLog(string logContent, Exception ex = null)
        {
            lock (lockObje)
            {
                try
                {
                    string path = Path.Combine(baseLogPath, ApplicationEx.ErrorLogDirectoryName, DateTime.Now.ToString("yyyyMMdd"));
                    if (!Directory.Exists(path))
                    {
                        Common.Utils.FileUtil.CreateDirectory(path);//创建新路径
                    }

                    string logFile = Path.Combine(path, DateTime.Now.ToString("yyyyMMddHH") + ".txt");
                    if (!File.Exists(logFile))//提出数据存放文件txt   
                    {
                        FileStream fs;
                        fs = File.Create(logFile);
                        fs.Close();
                    }
                    using (StreamWriter sw = File.AppendText(logFile))
                    {
                        sw.WriteLine("[" + System.DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.ffff") + "]" + Environment.NewLine + logContent);

                        if (ex != null)
                        {
                            sw.WriteLine("[" + ex.ToString() + "]" + Environment.NewLine + "详细信息：" + Environment.NewLine + ex.StackTrace);
                        }
                        sw.WriteLine("");
                        sw.Close();
                    }
                }
                catch { }
            }

        }

        public void RecordRunLog(string log)
        {
            lock (lockObje)
            {
                try
                {
                    string path = Path.Combine(baseLogPath, ApplicationEx.RunLogDirectoryName, DateTime.Now.ToString("yyyyMMdd"));
                    if (!Directory.Exists(path))
                    {
                        Directory.CreateDirectory(path);//创建新路径
                    }

                    string logFile = System.IO.Path.Combine(path, DateTime.Now.ToString("yyyyMMddHH") + ".txt");
                    if (!File.Exists(logFile))//提出数据存放文件txt   
                    {
                        FileStream fs;
                        fs = File.Create(logFile);
                        fs.Close();
                    }
                    using (StreamWriter sw = File.AppendText(logFile))
                    {

                        sw.WriteLine("[" + System.DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.ffff") + "]" + Environment.NewLine + log + "\r\n");
                        sw.WriteLine("");
                        sw.Close();
                    }
                }
                catch { }
            }
        }

        private void RecordFullCallStack(string message)
        {
            try
            {
                var stackTrace = new StackTrace(true);
                var frames = stackTrace.GetFrames();

                if (frames == null || frames.Length == 0)
                {
                    ElincloudServer.Common.Utils.LogUtil.RecordErrorLog($"{message} - 无法获取调用栈");
                    return;
                }

                var sb = new StringBuilder();
                sb.AppendLine($"===== {message} =====");

                // 跳过的类型关键词（可按需扩展）
                var skipTypes = new[] { "LogUtil", "CommonBL", "RecordFullCallStack" };

                // 从索引 2 开始：跳过 [0]=RecordFullCallStack, [1]=当前业务方法（如 GetServiceInfos）
                for (int i = 2; i < frames.Length; i++)
                {
                    var frame = frames[i];
                    if (frame == null) continue;

                    var method = frame.GetMethod();
                    if (method == null) continue;

                    string className = method.DeclaringType?.FullName ?? "UnknownType";
                    string methodName = method.Name;
                    string fileName = Path.GetFileName(frame.GetFileName() ?? "unknown");
                    int lineNumber = frame.GetFileLineNumber();

                    // 额外安全：如果跳过了太多导致没有有效帧，至少保留直接调用者（i=1）
                    // 此处我们坚持从 i=2 开始，因为 i=1 是被监控的方法本身

                    sb.AppendLine($"[{i - 2}] {className}.{methodName} ({fileName}:{lineNumber})");
                }

                // 如果从 i=2 开始没有内容，说明调用链太短（比如 Main 直接调用）
                if (sb.ToString().Count(c => c == '[') == 1) // 只有标题行
                {
                    // 回退到显示直接调用者（i=1）
                    sb.Clear();
                    sb.AppendLine($"===== {message} =====");
                    if (frames.Length > 1)
                    {
                        var frame = frames[1];
                        if (frame != null)
                        {
                            var method = frame.GetMethod();
                            string className = method?.DeclaringType?.FullName ?? "UnknownType";
                            string methodName = method?.Name ?? "Unknown";
                            string fileName = Path.GetFileName(frame.GetFileName() ?? "unknown");
                            int lineNumber = frame.GetFileLineNumber();
                            sb.AppendLine($"[0] {className}.{methodName} ({fileName}:{lineNumber})");
                        }
                    }
                    else
                    {
                        sb.AppendLine("[0] 调用来源未知");
                    }
                    sb.AppendLine("========================");
                }
                else
                {
                    sb.AppendLine("========================");
                }

                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog(sb.ToString());
            }
            catch
            {
                // 防止日志记录本身抛异常
            }
        }
    }
}
