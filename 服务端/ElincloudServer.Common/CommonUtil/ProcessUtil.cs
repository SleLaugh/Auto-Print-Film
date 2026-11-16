using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.AccessControl;
using System.Text;

public class ProcessUtil
{
    public bool CheckProcess(string processName)
    {
        try
        {
            // 检查进程是否响应（仅Windows）
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                int currentPid = Process.GetCurrentProcess().Id;

                var processes = Process.GetProcessesByName(processName);
                foreach (var process in processes)
                {
                    // 跳过自己
                    if (process.Id == currentPid || process.HasExited)
                        continue;

                    // 只要找到一个其他实例，就认为已运行
                    if (CheckProcessOnWindows(process))
                        return true;
                }
                return false; // 没有其他实例
            }
            else
            {
                // ✅ 获取当前 PID
                int currentPid = Process.GetCurrentProcess().Id;
                string dllFileName = $"{processName}.dll";

                // ✅ 传入 currentPid，让 GetPidByCmdline 跳过自己
                var pid = GetPidByCmdline(dllFileName, currentPid);
                if (pid == null) return false;

                using var process = Process.GetProcessById(pid.Value);
                return CheckProcessOnUnix(process);
            }
        }
        catch (Exception ex) when (ex is InvalidOperationException || ex is NotSupportedException)
        {
            // 进程可能在此期间退出，或权限不足
            return false;
        }
    }

    private bool CheckProcessOnWindows(Process process)
    {
        // 检查进程是否响应
        try
        {
            if (!process.Responding)
                return false;
        }
        catch
        {
            // 某些进程（如服务）可能不支持 Responding 属性
        }

        // 检查线程状态
        int totalThreads = 0;
        int suspendedThreads = 0;

        try
        {
            // 获取线程信息
            process.Refresh();

            foreach (ProcessThread thread in process.Threads)
            {
                totalThreads++;

                // 在Windows上，可以使用 ThreadState 和 WaitReason
                if (thread.ThreadState ==System.Diagnostics.ThreadState.Wait)
                {
                    // WaitReason 在 .NET Core Windows 上可用
                    if (thread.WaitReason == ThreadWaitReason.Suspended)
                    {
                        suspendedThreads++;
                    }
                }
            }

            // 如果只有一个线程，认为进程正常
            if (totalThreads <= 1)
                return true;

            // 如果除了一个线程外其他线程都挂起，认为进程可能卡死
            if (totalThreads > 1 && totalThreads == suspendedThreads + 1)
            {
                return false;
            }

            return true;
        }
        catch
        {
            // 如果无法获取线程信息，返回true（不判断为卡死）
            return true;
        }
    }

    private bool CheckProcessOnUnix(Process process)
    {

        // 检查进程是否在运行（通过/proc或ps命令）
        try
        {
            process.Refresh();
            return !process.HasExited;
        }
        catch
        {
            return false;
        }
    }

    // ✅ 关键改造：通过 /proc/pid/cmdline 匹配 DLL 文件名
    private static int? GetPidByCmdline(string dllFileName, int excludePid = -1)
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            return null;

        foreach (var dir in new DirectoryInfo("/proc").GetDirectories())
        {
            if (!int.TryParse(dir.Name, out int pid)) continue;

            // ✅ 跳过自己
            if (pid == excludePid) continue;

            try
            {
                string cmdlinePath = Path.Combine(dir.FullName, "cmdline");
                if (!File.Exists(cmdlinePath)) continue;

                byte[] bytes = File.ReadAllBytes(cmdlinePath);
                string cmdline = Encoding.UTF8.GetString(bytes).Replace("\0", " ");

                if (cmdline.Contains(dllFileName, StringComparison.OrdinalIgnoreCase))
                {
                    return pid;
                }
            }
            catch (UnauthorizedAccessException)
            {
                // 没权限读其他用户的进程，跳过
            }
            catch (IOException)
            {
                // 进程已退出
            }
            catch
            {
                // 其他异常忽略
            }
        }
        return null;
    }

    /// <summary>
    /// 仅在 Linux 上设置进程名（用于系统监视器显示），不影响 Windows。
    /// </summary>
    /// <param name="name">进程名，建议 ≤15 个 ASCII 字符</param>
    public void SetProcessNameForLinux(string name)
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            return;

        try
        {
            // Linux /proc/PID/comm 最大 15 字节（含结尾 \0）
            var safeName = name[..Math.Min(name.Length, 15)];
            var bytes = Encoding.ASCII.GetBytes(safeName + "\0");
            prctl(15, bytes); // PR_SET_NAME = 15
        }
        catch
        {
            // 安静失败，不影响主程序
        }
    }

    [DllImport("libc", SetLastError = true)]
    private static extern int prctl(int option, byte[] arg2);
}