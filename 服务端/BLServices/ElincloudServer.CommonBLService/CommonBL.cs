using ElincloudAISliceServer.DBEntity;
using ElincloudAISliceServer.Entity;
using ElincloudServer.Common;
using ElincloudServer.DBEntity;
using ElincloudServer.DBService;
using ElincloudServer.Entity;
using Microsoft.AspNetCore.WebUtilities;
using Microsoft.IdentityModel.Tokens;
using Microsoft.Win32;
using System.Collections.Concurrent;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Linq.Expressions;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Web;
using Module = ElincloudServer.DBEntity.Module;
using DicomMachineConfigInfo = ElincloudAISliceServer.Entity.DicomMachineConfigInfo;

namespace ElincloudServer.CommonBLService
{
    public class CommonBL
    {
        #region 独立服务相关

        private static MultiDictionary<string, string> configServerDics = null;

        public static MultiDictionary<string, string> ConfigServerDics
        {
            get
            {
                if (configServerDics == null)
                {
                    configServerDics = new MultiDictionary<string, string>();
                    configServerDics.Add("HospitalInfo", "ElincloudServer.AutoReferralService");
                    configServerDics.Add("HospitalInfo", "ElincloudServer.CloudImageSMSSendingService");
                    configServerDics.Add("HospitalInfo", "ElincloudServer.ImageDownloadEmailSendService");
                    configServerDics.Add("HospitalInfo", "ElincloudServer.ReportQualityService");
                    configServerDics.Add("CloudImageConfigInfo", "ElincloudServer.ReportQualityService");
                    configServerDics.Add("ServiceCommonConfigInfo", "ElincloudServer.ClearFileService");
                    configServerDics.Add("ServiceCommonConfigInfo", "ElincloudServer.CloudImageSMSSendingService");
                    configServerDics.Add("ServiceCommonConfigInfo", "ElincloudServer.ImageDownloadEmailSendService");
                    configServerDics.Add("ServiceCommonConfigInfo", "ElincloudServer.SeriesFileCaculateService");
                    configServerDics.Add("ServiceCommonConfigInfo", "ElincloudServer.CPUWarningService");
                    configServerDics.Add("ServiceCommonConfigInfo", "ElincloudServer.DiskWarningService");
                    configServerDics.Add("ServiceCommonConfigInfo", "ElincloudServer.ArchiveService");
                    configServerDics.Add("ServiceCommonConfigInfo", "ElincloudServer.CStoreService");
                    configServerDics.Add("ServiceCommonConfigInfo", "ElincloudServer.DataStoreService");
                    configServerDics.Add("ServiceCommonConfigInfo", "ElincloudServer.DBBackupService");
                    configServerDics.Add("ServiceCommonConfigInfo", "ElincloudServer.QueryRetrieveService");
                    configServerDics.Add("ServiceCommonConfigInfo", "ElincloudServer.ArchiveWarningService");
                    configServerDics.Add("ServiceCommonConfigInfo", "ElincloudServer.FileWebService");
                    configServerDics.Add("SMSConfigInfo", "ElincloudServer.CloudImageSMSSendingService");
                    configServerDics.Add("SMSConfigInfo", "ElincloudServer.AuthorizationWarningService");
                    configServerDics.Add("SMSConfigInfo", "ElincloudServer.CPUWarningService");
                    configServerDics.Add("SMSConfigInfo", "ElincloudServer.DicomUploadWarningService");
                    configServerDics.Add("SMSConfigInfo", "ElincloudServer.DiskWarningService");
                    configServerDics.Add("SMSConfigInfo", "ElincloudServer.FilmUploadWarningService");
                    configServerDics.Add("SMSConfigInfo", "ElincloudServer.ReportUploadWarningService");
                    configServerDics.Add("SMSConfigInfo", "ElincloudServer.ArchiveWarningService");
                    configServerDics.Add("EmailConfigInfo", "ElincloudServer.ImageDownloadEmailSendService");
                    configServerDics.Add("EmailConfigInfo", "ElincloudServer.AuthorizationWarningService");
                    configServerDics.Add("EmailConfigInfo", "ElincloudServer.CPUWarningService");
                    configServerDics.Add("EmailConfigInfo", "ElincloudServer.DicomUploadWarningService");
                    configServerDics.Add("EmailConfigInfo", "ElincloudServer.DiskWarningService");
                    configServerDics.Add("EmailConfigInfo", "ElincloudServer.FilmUploadWarningService");
                    configServerDics.Add("EmailConfigInfo", "ElincloudServer.ReportUploadWarningService");
                    configServerDics.Add("EmailConfigInfo", "ElincloudServer.ArchiveWarningService");
                    configServerDics.Add("ArchiveConfigInfo", "ElincloudServer.CloudImageSMSSendingService");
                    configServerDics.Add("ArchiveConfigInfo", "ElincloudServer.ArchiveService");
                    configServerDics.Add("ArchiveConfigInfo", "ElincloudServer.FileWebService");
                    configServerDics.Add("DataStoreConfigInfo", "ElincloudServer.ImageDownloadEmailSendService");
                    configServerDics.Add("DataStoreConfigInfo", "ElincloudServer.DataStoreService");
                    configServerDics.Add("FilmUploadWarningConfigInfo", "ElincloudServer.FilmUploadWarningService");
                    configServerDics.Add("AuthorizationWarningConfigInfo", "ElincloudServer.AuthorizationWarningService");
                    configServerDics.Add("DBBackupConfigInfo", "ElincloudServer.DBBackupService");
                    configServerDics.Add("CStoreConfigInfo", "ElincloudServer.CStoreService");
                    configServerDics.Add("ClearFileConfigInfo", "ElincloudServer.ClearFileService");
                    configServerDics.Add("SeriesFileCaculateConfigInfo", "ElincloudServer.SeriesFileCaculateService");
                    configServerDics.Add("DicomImageUploadWarningConfigInfo", "ElincloudServer.DicomUploadWarningService");
                    configServerDics.Add("CloudImageSMSSendingConfigInfo", "ElincloudServer.CloudImageSMSSendingService");
                    configServerDics.Add("CPUWarningConfigInfo", "ElincloudServer.CPUWarningService");
                    configServerDics.Add("DicomMachineConfigInfo", "ElincloudServer.CStoreService");
                    configServerDics.Add("QueryRetrieveConfigInfo", "ElincloudServer.QueryRetrieveService");
                    configServerDics.Add("WorklistConfigInfo", "ElincloudServer.WorklistService");
                    configServerDics.Add("ReportUploadWarningConfigInfo", "ElincloudServer.ReportUploadWarningService");
                    configServerDics.Add("DiskWarningConfigInfo", "ElincloudServer.DiskWarningService");
                    configServerDics.Add("CloudImageBIConfigInfo", "ElincloudServer.CloudImageBIService");
                    configServerDics.Add("ArchiveWarningConfigInfo", "ElincloudServer.ArchiveWarningService");
                    configServerDics.Add("RemoteDiagnosisBIConfigInfo", "ElincloudServer.RemoteDiagnosisBIService");
                    configServerDics.Add("ForwardingRunStatusWarningConfigInfo", "ElincloudServer.ForwardRunStatusWarningService");
                    configServerDics.Add("RemoteDepotStoreConfigInfo", "ElincloudServer.ObjectStorageService");
                    configServerDics.Add("AIDicomSliceConfig", "ElincloudAISliceServer.AISliceDicomService");
                }
                return configServerDics;
            }
        }

        public static void RestartServerByConfigName(string name)
        {
            ElincloudServer.Common.Utils.ThreadUtil.StartThread(new Action(() =>
            {
                if (ConfigServerDics.ContainsKey(name))
                {
                    List<string> ServiceNames = ConfigServerDics[name].ToList();
                    ServerConfigInfo configinfo = GetServerConfigInfo();
                    if (configinfo != null && configinfo.ServiceNames != null)
                    {
                        ServiceNames = ServiceNames.FindAll(p => configinfo.ServiceNames.Exists(c => c.ServiceENName == p));
                        foreach (string item in ServiceNames)
                        {
                            if (Common.Utils.ProcessUtil.CheckProcess(item))
                            {
                                RunServiceByCommand(item, ForwardingSystemCommandTypes.重启服务, configinfo.ServiceNames.FindAll(p => p.ServiceENName == item), "");
                                Task.Delay(500); //控制每个任务启动的节奏
                            }
                        }
                    }
                }
            }));
        }


        /// <summary>
        /// 获取服务列表信息
        /// </summary>
        /// <returns></returns>
        public static List<ServiceInfo> GetServiceInfos()
        {
            List<ServiceInfo> serviceInfos = new List<ServiceInfo>();
            serviceInfos.Add(new ServiceInfo() { ServiceCNName = "数据归档服务", Remark = "主要提供对远程检查设备传上来的DICOM图像归档功能！", ServiceENName = "ElincloudServer.ArchiveService", ServiceStatus = ForwardingSystemServiceStatus.未运行, ServiceShortName = "ArchiveSvc", ServiceRunLogPort = 9901 });
            serviceInfos.Add(new ServiceInfo() { ServiceCNName = "图像接收服务", Remark = "主要提供CStoreSCP服务来接收远程检查设备的DICOM图像！", ServiceENName = "ElincloudServer.CStoreService", ServiceStatus = ForwardingSystemServiceStatus.未运行, ServiceShortName = "CStoreSvc", ServiceRunLogPort = 9902 });
            serviceInfos.Add(new ServiceInfo() { ServiceCNName = "数据存储服务", Remark = "主要提供对上传到服务器的图像及视频数据自动存储到OSS或多级磁盘中！", ServiceENName = "ElincloudServer.DataStoreService", ServiceStatus = ForwardingSystemServiceStatus.未运行, ServiceShortName = "DataStoreSvc", ServiceRunLogPort = 9903 });
            serviceInfos.Add(new ServiceInfo() { ServiceCNName = "数据库备份服务", Remark = "主要提供对云影像的数据库进行每日自动备份并上传到对象存储服务中！", ServiceENName = "ElincloudServer.DBBackupService", ServiceStatus = ForwardingSystemServiceStatus.未运行, ServiceShortName = "DBBackupSvc", ServiceRunLogPort = 9904 });
            serviceInfos.Add(new ServiceInfo() { ServiceCNName = "文件清理服务", Remark = "主要定时清除日志文件、日志数据及临时缓存文件！", ServiceENName = "ElincloudServer.ClearFileService", ServiceStatus = ForwardingSystemServiceStatus.未运行, ServiceShortName = "ClearFileSvc", ServiceRunLogPort = 9905 });
            serviceInfos.Add(new ServiceInfo() { ServiceCNName = "序列归档状态更新服务", Remark = "主要用于更新影像接收完成状态！", ServiceENName = "ElincloudAISliceServer.ArchiveCompletenessService", ServiceStatus = ForwardingSystemServiceStatus.未运行, ServiceShortName = "ArchiveCompleteSvc", ServiceRunLogPort = 9925 });
            serviceInfos.Add(new ServiceInfo() { ServiceCNName = "智能排片切割影像服务", Remark = "主要用于为智能排片切割影像！", ServiceENName = "ElincloudAISliceServer.AISliceDicomService", ServiceStatus = ForwardingSystemServiceStatus.未运行, ServiceShortName = "AIFilmSliceSvc", ServiceRunLogPort = 9926 });
            return serviceInfos;
        }

        /// <summary>
        /// 运行服务命令
        /// </summary>
        /// <param name="serviceENName">服务名称</param>
        /// <param name="commandTypes">服务命令</param>
        /// <param name="ServiceInfos">服务列表信息</param>
        /// <param name="args">参数</param>
        public static void RunServiceByCommand(string serviceENName, ForwardingSystemCommandTypes commandTypes, List<ServiceInfo> ServiceInfos, string args = "")
        {
            try
            {
                List<ServiceInfo> serviceIinfos = ServiceInfos;
                ServiceInfo service = serviceIinfos.FirstOrDefault(x => x.ServiceENName == serviceENName);
                if (service != null)
                {
                    switch (commandTypes)
                    {
                        case ForwardingSystemCommandTypes.启动服务:
                            StartService(service.ServiceENName, args);
                            break;
                        case ForwardingSystemCommandTypes.停止服务:
                            StopService(service);
                            break;
                        case ForwardingSystemCommandTypes.重启服务:
                            StopService(service);
                            StartService(service.ServiceENName, args);
                            break;
                    }
                }
                else
                {
                    new ServiceException("在服务器未找到对应的服务！");
                }
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("启动独立服务异常！", ex);
            }
        }

        private static void StartService(string ServiceENName, string args = "")
        {
            if (ElincloudServer.Common.Utils.ConfigUtil.IsWindow)
            {
                if (!Utils.ProcessUtil.CheckProcess(ServiceENName))
                {
                    if (ServiceENName == "ElincloudServer.ObjectStorageService" || ServiceENName == "ElincloudServer.ArchiveService")
                    {
                        StartDataStoreService(ServiceENName);
                    }
                    else
                    {
                        ProcessStartInfo psi = new ProcessStartInfo
                        {
                            FileName = Utils.StringUtil.Combine(AppContext.BaseDirectory, ServiceENName + ".exe"),
                            Arguments = args, //
                            UseShellExecute = true, // 允许 Windows 以新窗口启动进程
                        };
                        Process.Start(psi);
                    }
                }
            }
            else
            {
                if (!Utils.ProcessUtil.CheckProcess(ServiceENName))
                {
                    string shellPath = ExecuteCommandWithOutput("bash", "-lc \"echo $PATH\"");
                    //查找dotnet执行命令的目录 
                    string dotnetPath = ExecuteCommandWithOutput("which", "dotnet", shellPath);
                    string fileName = Utils.StringUtil.Combine(AppContext.BaseDirectory, ServiceENName + ".dll");
                    ExecuteCommand(dotnetPath, $"\"{fileName}\" {args}");
                }
            }
        }
        private static void StartDataStoreService(string ServiceENName, string args = "")
        {
            string exePath = Utils.StringUtil.Combine(AppContext.BaseDirectory, ServiceENName + ".exe");
            string taskName = $"IIS_Launch_{ServiceENName}";

            // 1. 获取当前的交互式用户（比如 Administrator）
            string userName = System.Security.Principal.WindowsIdentity.GetCurrent().Name;

            // 2. 创建计划任务，指定在交互式用户的会话中运行（Session 1）
            string createCommand = $@"/create /tn ""{taskName}"" /tr ""{exePath} {args}"" /sc once /st 00:00 /ru ""{userName}"" /rl HIGHEST /f";

            ProcessStartInfo psi = new ProcessStartInfo
            {
                FileName = "schtasks",
                Arguments = createCommand,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                CreateNoWindow = true
            };

            using (var proc = Process.Start(psi))
            {
                proc.WaitForExit();
                if (proc.ExitCode == 0)
                {
                    // 3. 立即运行任务（这会在用户的交互式桌面启动进程）
                    Process.Start("schtasks", $"/run /tn \"{taskName}\"");
                    //Utils.LogUtil.RecordErrorLog($"服务已在交互式会话中启动: {ServiceENName}");
                }
                else
                {
                    Utils.LogUtil.RecordErrorLog($"创建任务失败，退出码: {proc.ExitCode}");
                }
            }
        }
        private static void StopService(ServiceInfo service, bool isAuthService = true)
        {
            if (ElincloudServer.Common.Utils.ConfigUtil.IsWindow)
            {
                foreach (var process in Process.GetProcessesByName(service.ServiceENName))
                {
                    if (!process.CloseMainWindow())
                    {
                        process.Kill(true); // 强制关闭进程
                    }
                    process.WaitForExit();
                }
                //循环关闭 关联服务
                if (service.RelatedServices != null && service.RelatedServices.Count > 0)
                {
                    foreach (var item in service.RelatedServices)
                    {
                        foreach (var process in Process.GetProcessesByName(item.ServiceENName))
                        {
                            if (!process.CloseMainWindow())
                            {
                                process.Kill(true); // 强制关闭进程
                            }
                            process.WaitForExit();
                        }
                    }
                }
            }
            else
            {
                // 停止服务
                ExecuteCommand("systemctl", $"stop {service.ServiceENName}");

                ExecuteCommand("pkill", $"-9 -f {service.ServiceENName}.dll");
                //循环关闭 关联服务
                if (service.RelatedServices != null && service.RelatedServices.Count > 0)
                {
                    foreach (var item in service.RelatedServices)
                    {
                        ExecuteCommand("pkill", $"-9 -f {item.ServiceENName}.dll");
                    }
                }
            }
            GC.Collect();
            Thread.Sleep(2000); // 等待资源释放,不然Utils.ProcessUtil.CheckProcess方法会判断程序还在运行中
        }

        /// <summary>
        /// 设置服务自启配置
        /// </summary>
        /// <param name="service"></param>
        /// <returns></returns>
        public static bool SetAotuStartService(ServiceInfo service, bool isAotuStart)
        {
            try
            {
                if (ElincloudServer.Common.Utils.ConfigUtil.IsWindow)
                {
                    RegistryKey R_local = Registry.LocalMachine;//RegistryKey R_local = Registry.CurrentUser;
                    string subPath = Environment.Is64BitOperatingSystem ? "\\WOW6432Node" : "";
                    RegistryKey R_run = Registry.LocalMachine;
                    if (isAotuStart)
                    {
                        R_run = R_local.CreateSubKey(@"SOFTWARE" + subPath + @"\Microsoft\Windows\CurrentVersion\Run");
                        R_run.SetValue(service.ServiceENName, Utils.StringUtil.Combine(AppContext.BaseDirectory, service.ServiceENName + ".exe") + " " + "autorun");
                    }
                    else
                    {
                        R_run = R_local.OpenSubKey(@"SOFTWARE" + subPath + @"\Microsoft\Windows\CurrentVersion\Run", true);
                        if (R_run != null)
                        {
                            // 检查键值是否存在后删除
                            if (R_run.GetValue(service.ServiceENName) != null)
                            {
                                R_run.DeleteValue(service.ServiceENName);
                            }
                        }
                    }
                    R_run.Close();
                    R_local.Close();
                    return true;
                }
                else
                {
                    if (isAotuStart)
                    {
                        string serviceFilePath = $"/etc/systemd/system/{service.ServiceENName}.service";
                        string shellPath = ExecuteCommandWithOutput("bash", "-lc \"echo $PATH\"");
                        //查找dotnet执行命令的目录 
                        string dotnetPath = ExecuteCommandWithOutput("which", "dotnet", shellPath);
                        string dllFullPath = Utils.StringUtil.Combine(AppContext.BaseDirectory, service.ServiceENName + ".dll");

                        // 确保路径格式，Linux 路径分隔符是 `/`
                        dllFullPath = dllFullPath.Replace('\\', '/');
                        if (File.Exists(serviceFilePath))
                        {
                            //设置开机自启动
                            ExecuteCommand("systemctl", $"enable  {service.ServiceENName}");
                            return true;
                        }

                        string serviceContent = $@"
                                            [Unit]
                                            Description={AppContext.BaseDirectory}
                                            After=network.target
                    
                                            [Service]
                                            ExecStart={dotnetPath} {dllFullPath}
                                            WorkingDirectory={AppContext.BaseDirectory}
                                            Restart=always
                                            User=root
                                            Group=root
                    
                                            [Install]
                                            WantedBy=multi-user.target
                                            ";
                        File.WriteAllText(serviceFilePath, serviceContent);
                        //设置开机自启动
                        ExecuteCommand("systemctl", $"enable  {service.ServiceENName}");
                    }
                    else
                    {
                        // 禁用服务
                        ExecuteCommand("systemctl", $"disable {service.ServiceENName}");
                        // 停止服务
                        ExecuteCommand("systemctl", $"stop {service.ServiceENName}");
                    }
                    return true;

                }
            }
            catch (Exception ex)
            {
                Common.Utils.LogUtil.RecordErrorLog("设置服务[" + service.ServiceCNName + "]自动动时异常", ex);
                return false;
            }
        }

        public static void ExecuteCommand(string command, string arguments)
        {
            try
            {
                ProcessStartInfo psi = new ProcessStartInfo
                {
                    FileName = command,
                    Arguments = arguments,
                    UseShellExecute = true,
                    CreateNoWindow = true,
                    RedirectStandardOutput = false,
                    RedirectStandardError = false
                };
                var process = Process.Start(psi);

                if (process == null)
                {
                    ElincloudServer.Common.Utils.LogUtil.RecordErrorLog($"Linux 启动失败：dotnet {command} 返回 null");
                }

            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog($"执行命令失败: {ex.Message} 命令内容{command}");
            }
        }

        public static string ExecuteCommandWithOutput(string command, string arguments, string mergedPath = "")
        {
            try
            {
                ProcessStartInfo psi = new ProcessStartInfo
                {
                    FileName = command,
                    Arguments = arguments,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };
                //添加环境变量
                if (!string.IsNullOrEmpty(mergedPath))
                {
                    psi.EnvironmentVariables["PATH"] = mergedPath;
                }
                using (var process = Process.Start(psi))
                {
                    var outputTask = process.StandardOutput.ReadToEndAsync();
                    var errorTask = process.StandardError.ReadToEndAsync();
                    process.WaitForExit();

                    string output = outputTask.Result.Trim();
                    string error = errorTask.Result.Trim();

                    if (!string.IsNullOrEmpty(error))
                    {
                        ElincloudServer.Common.Utils.LogUtil.RecordErrorLog($"执行命令失败: {error}");
                    }

                    return output;
                }
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog($"执行命令异常: {ex.Message}");
                return string.Empty;
            }
        }
        #endregion

        #region 获取需要启动的服务列表

        public static ServiceInfo GetServerInfoByServerENName(string serviceENName)
        {
            if (string.IsNullOrWhiteSpace(serviceENName))
            {
                return new ServiceInfo();
            }
            else
            {
                ServerConfigInfo serverConfigInfo = GetServerConfigInfo();
                if (serverConfigInfo != null && serverConfigInfo.ServiceNames.Exists(p => p.ServiceENName == serviceENName))
                {
                    return serverConfigInfo.ServiceNames.FirstOrDefault(p => p.ServiceENName == serviceENName);
                }
                else
                {
                    return GetServiceInfos().FirstOrDefault(p => p.ServiceENName == serviceENName);
                }
            }
        }

        private static ServerConfigInfo serverConfigInfo = null;

        public static ServerConfigInfo GetServerConfigInfo(bool updateCache = false)
        {
            if (serverConfigInfo == null || updateCache)
            {
                List<ServerConfig> infos = DBServices.ElincloudConfigInstance.GetMuch<ServerConfig>();
                if (infos != null && infos.Count > 0)
                {
                    ServerConfigInfo info = ElincloudServer.Common.Utils.JsonUtil.FromJson<ServerConfigInfo>(infos.FirstOrDefault().ServiceNameConfig);
                    return info;
                }
                else
                {
                    return new ServerConfigInfo();
                }
            }
            return serverConfigInfo;
        }

        #endregion

        #region 配置相关

        #region 公共配置

        private static ServiceCommonConfigInfo serviceCommonConfigInfo = null;
        public static ServiceCommonConfigInfo GetServiceCommonConfigInfo(bool updateCache = false)
        {
            if (serviceCommonConfigInfo == null || updateCache)
            {
                List<ServiceCommonConfig> commonConfigs = DBServices.ElincloudConfigInstance.GetMuch<ServiceCommonConfig>();
                if (commonConfigs.Count > 0)
                {
                    ServiceCommonConfig commonConfig = commonConfigs.FirstOrDefault();
                    serviceCommonConfigInfo = ElincloudServer.Common.Utils.JsonUtil.FromJson<ServiceCommonConfigInfo>(commonConfig.CommonConfig);
                }
                else
                {
                    return new ServiceCommonConfigInfo();
                }
            }
            return serviceCommonConfigInfo;
        }

        #endregion

        #region Dicom设备通信配置缓存

        private static readonly object dicomMachineConfigLock = new object();

        // 以数据库 HospitalCode 为键。
        // 字典发布后不再直接修改，刷新时构建新字典并替换。
        private static Dictionary<string, DicomMachineConfigInfo> dicomMachineConfigCache = null;

        /// <summary>
        /// 获取全部医院的设备配置。
        /// updateCache 为 true 时，重新加载全部配置。
        /// </summary>
        public static List<DicomMachineConfigInfo> GetDicomMachineInfos(bool updateCache = false)
        {
            Dictionary<string, DicomMachineConfigInfo> cache = GetDicomMachineConfigCache(updateCache);
            return cache.Values.ToList();
        }

        /// <summary>
        /// 根据医院 Code 获取设备配置。
        /// updateCache 为 true 时，只刷新指定医院的配置。
        /// 未找到时返回空配置对象。
        /// </summary>
        public static DicomMachineConfigInfo GetDicomMachineInfoByHospitalCode(string hospitalCode, bool updateCache = false)
        {
            if (string.IsNullOrEmpty(hospitalCode))
            {
                return new DicomMachineConfigInfo();
            }

            if (updateCache)
            {
                RefreshDicomMachineConfigCache(hospitalCode);
            }

            Dictionary<string, DicomMachineConfigInfo> cache = GetDicomMachineConfigCache();

            DicomMachineConfigInfo info;
            if (cache.TryGetValue(hospitalCode, out info))
            {
                return info;
            }

            return new DicomMachineConfigInfo();
        }

        /// <summary>
        /// 刷新指定医院的设备配置缓存。
        /// 缓存尚未初始化时，先加载全部医院配置。
        /// </summary>
        public static void RefreshDicomMachineConfigCache(string hospitalCode)
        {
            if (string.IsNullOrEmpty(hospitalCode))
            {
                return;
            }

            lock (dicomMachineConfigLock)
            {
                if (dicomMachineConfigCache == null)
                {
                    // 首次加载全部配置，其中已包含当前医院的最新配置。
                    // C# 的 lock 支持同一线程重入。
                    GetDicomMachineConfigCache();
                    return;
                }

                DicomMachineConfig config = DBServices.ElincloudConfigInstance.GetOne<DicomMachineConfig>(p => p.HospitalCode == hospitalCode);

                DicomMachineConfigInfo info = null;

                if (config != null)
                {
                    info = ElincloudServer.Common.Utils.JsonUtil.FromJson<DicomMachineConfigInfo>(config.Machines);
                }

                // 复制现有字典，保留其他医院的缓存。
                // 不修改旧字典，保证正在读取旧缓存的线程可继续读取。
                Dictionary<string, DicomMachineConfigInfo> newCache = new Dictionary<string, DicomMachineConfigInfo>(dicomMachineConfigCache, StringComparer.Ordinal);

                if (info != null)
                {
                    newCache[hospitalCode] = info;
                }
                else
                {
                    // 数据库中已不存在该配置，或反序列化结果为空。
                    newCache.Remove(hospitalCode);
                }

                dicomMachineConfigCache = newCache;
            }
        }

        /// <summary>
        /// 获取内部缓存。
        /// 首次访问或指定全量刷新时，从数据库加载全部配置。
        /// </summary>
        private static Dictionary<string, DicomMachineConfigInfo> GetDicomMachineConfigCache(bool updateCache = false)
        {
            lock (dicomMachineConfigLock)
            {
                if (dicomMachineConfigCache == null || updateCache)
                {
                    List<DicomMachineConfig> configs = DBServices.ElincloudConfigInstance.GetMuch<DicomMachineConfig>();

                    Dictionary<string, DicomMachineConfigInfo> newCache = new Dictionary<string, DicomMachineConfigInfo>(StringComparer.Ordinal);

                    if (configs != null)
                    {
                        foreach (DicomMachineConfig config in configs)
                        {
                            if (config == null || string.IsNullOrEmpty(config.HospitalCode))
                            {
                                continue;
                            }

                            DicomMachineConfigInfo info = ElincloudServer.Common.Utils.JsonUtil.FromJson<DicomMachineConfigInfo>(config.Machines);

                            if (info != null)
                            {
                                newCache[config.HospitalCode] = info;
                            }
                        }
                    }

                    // 全部加载完成后再替换。
                    // 空结果也缓存；加载抛出异常时保留旧缓存。
                    dicomMachineConfigCache = newCache;
                }

                return dicomMachineConfigCache;
            }
        }

        #endregion

        #region 获取归档配置

        private static ConcurrentDictionary<string, ArchiveConfigInfo> archiveConfigInfoDic = new ConcurrentDictionary<string, ArchiveConfigInfo>();
        public static ArchiveConfigInfo GetArchiveConfigInfoByHospitalCode(string hospitalCode, bool updateCache = false)
        {
            if (!archiveConfigInfoDic.ContainsKey(hospitalCode) || updateCache)
            {
                ArchiveConfig config = DBServices.ElincloudConfigInstance.GetOne<ArchiveConfig>(p => p.HospitalCode == hospitalCode);
                ArchiveConfigInfo info;
                if (config != null)
                {
                    info = ElincloudServer.Common.Utils.JsonUtil.FromJson<ArchiveConfigInfo>(config.Archive);
                }
                else
                {
                    info = new ArchiveConfigInfo { ArchivePathRules = ArchivePathRules.Default };
                }
                archiveConfigInfoDic.AddOrUpdate(hospitalCode, info, (hospitalCode, archiveConfigInfo) =>
                {
                    archiveConfigInfo = info;
                    return archiveConfigInfo;
                });
            }
            return archiveConfigInfoDic[hospitalCode];
        }

        #endregion        

        #region 数据存储配置

        private static DataStoreConfigInfo dataStoreConfigInfo = null;
        public static DataStoreConfigInfo GetDataStoreConfigInfo(bool updateCache = false)
        {
            if (dataStoreConfigInfo == null || updateCache)
            {
                List<DataStoreConfig> infos = DBServices.ElincloudConfigInstance.GetMuch<DataStoreConfig>();
                if (infos.Count > 0)
                {
                    DataStoreConfig config = infos.FirstOrDefault();
                    dataStoreConfigInfo = ElincloudServer.Common.Utils.JsonUtil.FromJson<DataStoreConfigInfo>(config.DataStore);
                }
                else
                {
                    return new DataStoreConfigInfo();
                }
            }
            return dataStoreConfigInfo;
        }

        #endregion

        #region 清除文件配置
        public static ClearFileConfigInfo GetClearFileConfigInfo()
        {
            List<ClearFileConfig> infos = DBServices.ElincloudConfigInstance.GetMuch<ClearFileConfig>();
            if (infos.Count > 0)
            {
                ClearFileConfig config = infos.FirstOrDefault();
                ClearFileConfigInfo info = ElincloudServer.Common.Utils.JsonUtil.FromJson<ClearFileConfigInfo>(config.ClearFile);
                return info;
            }
            else
            {
                return new ClearFileConfigInfo();
            }
        }

        #endregion

        #region 数据库备份服务
        public static DBBackupConfigInfo GetDBBackupConfigInfo()
        {
            List<DBBackupConfig> infos = DBServices.ElincloudConfigInstance.GetMuch<DBBackupConfig>();
            if (infos.Count > 0)
            {
                DBBackupConfig config = infos.FirstOrDefault();
                DBBackupConfigInfo info = ElincloudServer.Common.Utils.JsonUtil.FromJson<DBBackupConfigInfo>(config.DBBackup);
                return info;
            }
            else
            {
                return new DBBackupConfigInfo();
            }
        }
        #endregion

        #region 智能切片配置
        public static AIDicomSliceConfigInfo GetAIDicomSliceConfigInfo()
        {
            List<AIDicomSliceConfig> infos = DBServices.ElincloudConfigInstance.GetMuch<AIDicomSliceConfig>();
            if (infos.Count > 0)
            {
                AIDicomSliceConfig config = infos.FirstOrDefault();
                AIDicomSliceConfigInfo info = ElincloudServer.Common.Utils.JsonUtil.FromJson<AIDicomSliceConfigInfo>(config.AIDicomSlice);
                info.Initialize();
                return info;
            }
            else
            {
                return new AIDicomSliceConfigInfo();
            }
        }
        #endregion

        #region 切片方案配置
        /// <summary>
        /// 根据ID获取切片方案
        /// </summary>
        /// <param name="hospitalGroupID"></param>
        /// <returns></returns>
        public static SlicePlanConfigInfo GetSlicePlanConfigInfoByID(string hospitalGroupID, string id)
        {
            SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
            SlicePlanConfig entity = instance.GetOne<SlicePlanConfig>(i => i.ID == id);
            if (entity == null)
            {
                throw new ServiceException("未找到对应的切片方案");
            }
            SlicePlanConfigInfo result = MapperSlicePlanConfigInfo(entity);
            return result;
        }

        /// <summary>
        /// 获取切片方案列表
        /// </summary>
        /// <param name="hospitalGroupID"></param>
        /// <returns></returns>
        public static List<SlicePlanConfigInfo> GetSlicePlanConfigInfos(string hospitalGroupID)
        {
            AIDicomSliceConfigInfo config = GetAIDicomSliceConfigInfo();

            List<SlicePlanConfigInfo> ret = new List<SlicePlanConfigInfo>();
            SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
            List<SlicePlanConfig> list = instance.GetMuch<SlicePlanConfig>();
            ret = list.Select(i =>
            {
                SlicePlanConfigInfo info = MapperSlicePlanConfigInfo(i);
                SliceRuleConfigInfo rule = config.SliceRuleConfigInfos.FirstOrDefault(t => t.StudyType == info.StudyTypeName && t.SlicePart == info.SlicePartName && t.Plane == info.Plane && t.SliceType == info.SliceType);
                List<SliceRegions> regions = rule != null ? rule.Regions : new List<SliceRegions>() { SliceRegions.Default };
                info.Regions = regions;
                return info;
            }).ToList();
            return ret;
        }

        public static SlicePlanConfigInfo MapperSlicePlanConfigInfo(SlicePlanConfig entity)
        {
            SlicePlanConfigInfo result = new SlicePlanConfigInfo()
            {
                ID = entity.ID,
                Name = entity.Name,
                IsEnabel = entity.IsEnabel,
                StudyTypeName = entity.StudyTypeName,
                StudyPartName = entity.StudyPartName,
                MinScanThickness = entity.MinScanThickness,
                MaxScanThickness = entity.MaxScanThickness,

                SlicePartName = entity.SlicePartName,
                Plane = ElincloudServer.Common.Utils.EnumUtil.GetEnumValue<PlaneEnums>(entity.Plane),
                SliceType = ElincloudServer.Common.Utils.EnumUtil.GetEnumValue<SliceTypes>(entity.SliceType),
                SliceThickness = entity.SliceThickness,
                SliceSpacing = entity.SliceSpacing,
                SliceImageCount = entity.SliceImageCount,
                FillPercent = entity.FillPercent,
                WindowPreset = entity.WindowPreset,
            };
            return result;
        }
        #endregion

        #region AI切割预设配置

        public static AIDicomSlicePresetConfigInfo GetAIDicomSlicePresetConfigInfo(string hospitalCode)
        {
            AIDicomSlicePresetConfigInfo result = new AIDicomSlicePresetConfigInfo();
            AIDicomSlicePresetConfig config = DBServices.ElincloudConfigInstance.GetOne<AIDicomSlicePresetConfig>(i => i.HospitalCode == hospitalCode);
            if (config != null)
            {
                result = ElincloudServer.Common.Utils.JsonUtil.FromJson<AIDicomSlicePresetConfigInfo>(config.AIDicomSlicePreset);
            }
            else
            {
                result = new AIDicomSlicePresetConfigInfo();
                result.Initialize();
                return result;
            }
            return result;
        }

        #endregion

        #region AI切割结果推送配置

        public static AISliceResultPushConfigInfo GetAISliceResultPushConfigInfo(string hospitalCode)
        {
            AISliceResultPushConfigInfo result = new AISliceResultPushConfigInfo();
            AISliceResultPushConfig config = DBServices.ElincloudConfigInstance.GetOne<AISliceResultPushConfig>(i => i.HospitalCode == hospitalCode);
            if (config != null)
            {
                result = ElincloudServer.Common.Utils.JsonUtil.FromJson<AISliceResultPushConfigInfo>(config.AISliceResultPush);
            }
            else
            {
                return new AISliceResultPushConfigInfo();
            }
            return result;
        }

        #endregion

        #region 密码强度验证配置
        public static PasswordPolicyConfigInfo GetPasswordPolicyConfigInfoByHospitalCode(string hospitalCode)
        {
            PasswordPolicyConfig config = DBServices.ElincloudConfigInstance.GetOne<PasswordPolicyConfig>(p => p.HospitalCode == hospitalCode);
            if (config != null)
            {
                PasswordPolicyConfigInfo info = ElincloudServer.Common.Utils.JsonUtil.FromJson<PasswordPolicyConfigInfo>(config.PasswordPolicy);
                return info;
            }
            else
            {
                return new PasswordPolicyConfigInfo();
            }
        }

        /// <summary>
        /// 密码强度验证
        /// </summary>
        /// <param name="hospitalCode"></param>
        /// <remarks>
        /// sle 2026年1月14日14:08:31
        /// </remarks>
        public static void PasswordPolicyVerification(string hospitalCode, string password)
        {
            PasswordPolicyConfigInfo config = GetPasswordPolicyConfigInfoByHospitalCode(hospitalCode);
            password = Utils.StringUtil.Decrypt(password);

            // 配置不存在时，校验失败
            if (config == null)
            {
                throw new ServiceException("W00000466");
            }

            // 1. 长度校验：当MinLength和MaxLength都为0时，跳过校验
            //    如果其中一个为0，则MinLength视为0，MaxLength视为无限大
            if (!(config.MinLength == 0 && config.MaxLength == 0))
            {
                int minLength = config.MinLength > 0 ? config.MinLength : 0;
                int maxLength = config.MaxLength > 0 ? config.MaxLength : int.MaxValue;

                // 密码长度必须在[minLength, maxLength]区间内
                if (password.Length < minLength || password.Length > maxLength)
                {
                    throw new ServiceException("W00000467");
                }
            }

            // 2. 小写字母校验：如开启此项，则密码必须至少包含一个小写字母
            bool hasLower = password.Any(c => char.IsLower(c));
            if (config.RequireLowercase && !hasLower)
            {
                throw new ServiceException("W00000468");
            }
            // 3. 大写字母校验：如开启此项，则密码必须至少包含一个大写字母
            bool hasUpper = password.Any(c => char.IsUpper(c));
            if (config.RequireUppercase && !hasUpper)
            {
                throw new ServiceException("W00000469");
            }

            // 4. 数字校验：如开启此项，则密码必须至少包含一个数字
            bool hasDigit = password.Any(c => char.IsDigit(c));
            if (config.RequireDigit && !hasDigit)
            {
                throw new ServiceException("W00000470");
            }
            // 5. 特殊字符校验：如开启此项，则密码必须至少包含一个指定特殊字符
            string specialChars = "!@#$%^&*()-=+[]{}|;:,.<>?/`~";
            bool hasSpecialChar = password.Any(c => specialChars.Contains(c));
            if (config.RequireSpecialChar && !hasSpecialChar)
            {
                throw new ServiceException("W00000471");
            }

            // 6. 下划线校验：如开启此项，则密码必须至少包含一个下划线
            bool hasUnderscore = password.Contains('_');
            if (config.RequireUnderscore && !hasUnderscore)
            {
                throw new ServiceException("W00000472");
            }
        }

        #endregion

        #endregion

        #region 角色相关

        /// <summary>
        /// 根据用户ID获取该用户所关联的角色列表
        /// </summary>
        /// <param name="userID">用户ID</param>
        /// <returns>角色列表</returns>
        public static List<RoleInfo> GetUserRolesByUserID(SqlAccess instance, string userID, List<DoctorRoleRela> doctorRoleRelas = null, List<Role> roleList = null)
        {
            List<RoleInfo> roleInfos = new List<RoleInfo>();
            List<DoctorRoleRela> userRoles = null;
            //查询用户角色关系信息
            if (doctorRoleRelas != null)
            {
                userRoles = doctorRoleRelas.FindAll(c => c.DoctorID == userID);
            }
            else
            {
                userRoles = instance.GetMuch<DoctorRoleRela>(o => o.DoctorID == userID);
            }

            if (userRoles != null && userRoles.Count > 0)
            {
                List<string> roleIDs = userRoles.Select(c => c.RoleID).ToList();
                List<Role> roles = null;
                if (roleList != null)
                {
                    roles = roleList.FindAll(c => roleIDs.Contains(c.ID));
                }
                else
                {
                    roles = instance.GetMuch<Role>(p => roleIDs.Contains(p.ID));
                }

                if (roles != null)
                {
                    //循环组装角色列表结果集
                    foreach (var role in roles)
                    {
                        roleInfos.Add(new RoleInfo()
                        {
                            ID = role.ID,
                            Name = role.Name,
                            Remark = role.Remark,
                        });
                    }
                }
            }
            return roleInfos;
        }

        #endregion

        #region 组织机构相关

        public static OrganizationInfo GetUserOrganizationByUserID(SqlAccess instance, string userid, DoctorOrganizationRela doRela = null, List<Organization> orgs = null, List<Address> adds = null)
        {
            DoctorOrganizationRela docOrgRela = doRela != null ? doRela : instance.GetOne<DoctorOrganizationRela>(o => o.DoctorID == userid); ;
            if (docOrgRela != null)
            {
                //获取用户关联的组织机构信息
                Organization org = orgs != null ? orgs.FirstOrDefault(c => c.ID == docOrgRela.OrganizationID) : instance.GetOne<Organization>(p => p.ID == docOrgRela.OrganizationID);
                if (org != null)
                {
                    return new OrganizationInfo()
                    {
                        ID = org.ID,
                        PID = org.PID,
                        Name = org.Name,
                        Code = org.Code,
                        Remark = org.Remark,
                    };
                }
            }
            return null;
        }

        public static HospitalInfo GetHospitalInfoByID(string hospitalID)
        {
            Hospital hospital = DBServices.ElincloudBasicInstance.GetOne<Hospital>(p => p.ID == hospitalID);
            if (hospital != null)
            {
                return new HospitalInfo()
                {
                    ID = hospital.ID,
                    Name = hospital.Name,
                    EasyName = hospital.EasyName,
                    Code = hospital.Code,
                    Remark = hospital.Remark,
                    SortNumber = hospital.SortNumber,
                    Address = hospital.Address,
                    Longitude = hospital.Longitude,
                    Latitude = hospital.Latitude,
                    HospitalGroupID = hospital.HospitalGroupID,
                    HospitalGroupName = hospital.HospitalGroupName
                };
            }
            else
            {
                throw new ServiceException("未获取到ID为[" + hospitalID + "]的医院信息！");
            }
        }

        public static List<HospitalInfo> GetHospitalInfos()
        {
            List<HospitalInfo> rets = new List<HospitalInfo>();
            List<Hospital> hospitals = DBServices.ElincloudBasicInstance.GetMuch<Hospital>();
            if (hospitals != null && hospitals.Count > 0)
            {
                foreach (var hospital in hospitals)
                {
                    rets.Add(new HospitalInfo()
                    {
                        ID = hospital.ID,
                        Name = hospital.Name,
                        Remark = hospital.Remark,
                        Code = hospital.Code,
                        EasyName = hospital.EasyName,
                        SortNumber = hospital.SortNumber,
                        Address = hospital.Address,
                        Longitude = hospital.Longitude,
                        Latitude = hospital.Latitude,
                        HospitalGroupID = hospital.HospitalGroupID,
                        HospitalGroupName = hospital.HospitalGroupName,
                        AreaID = hospital.AreaID,
                        AreaName = hospital.AreaName,
                        CityID = hospital.CityID,
                        CityName = hospital.CityName,
                        HospitalFreeTypes = hospital.HospitalFreeTypes,
                        HospitalRegionTypes = hospital.HospitalRegionTypes,
                        HospitalSizes = hospital.HospitalSizes,
                        ProvinceID = hospital.ProvinceID,
                        ProvinceName = hospital.ProvinceName
                    });
                }
            }
            return rets.OrderBy(x => x.SortNumber).ToList();
        }

        public static List<OrganizationInfo> GetSubOrganizations(string parentOrgID, List<Organization> allOrgs)
        {
            List<OrganizationInfo> result = new List<OrganizationInfo>();
            if (allOrgs != null)
            {
                List<Organization> subOrgs = allOrgs.FindAll(p => p.PID == parentOrgID).ToList();
                if (subOrgs != null && subOrgs.Count() > 0)
                {
                    foreach (var item in subOrgs)
                    {
                        result.Add(new OrganizationInfo()
                        {
                            ID = item.ID,
                            PID = item.PID,
                            Name = item.Name,
                            Code = item.Code,
                            Remark = item.Remark,
                        });
                        List<OrganizationInfo> infos = GetSubOrganizations(item.ID, allOrgs);
                        result.AddRange(infos);
                    }

                }
            }
            return result;
        }

        #endregion

        #region 模块相关

        /// <summary>
        /// 获取当前模块节点下所有子节点
        /// </summary>
        /// <param name="module"></param>
        /// <returns></returns>
        public static List<Module> GetChildrenModulesByModule(ModuleInfo moduleInfo)
        {

            if (moduleInfo != null)
            {
                var modules = DBServices.ElincloudBasicInstance.GetMuch<Module>();
                if (modules != null && modules.Count > 0)
                {
                    Module pmodule = modules.FirstOrDefault(p => p.ID == moduleInfo.ID);
                    List<Module> result = RecursionModuleChild(pmodule, modules);
                    if (result != null)
                    {
                        return result;
                    }
                    else
                    {
                        throw new ServiceException("没有获取到模块ID[" + pmodule.ID + "]的子模块信息！");
                    }
                }
                else
                {
                    throw new ServiceException("没有获取到模块信息!");
                }
            }
            else
            {
                throw new ServiceException("上级模块信息不能为空！");
            }

        }
        /// <summary>
        /// 迭代返回当前模块及所有下级模块信息
        /// </summary>
        /// <returns></returns>
        private static List<Module> RecursionModuleChild(Module parent, List<Module> modules)
        {
            return ExcuteUtil.ExcuteOrDefault<List<Module>>(new Func<List<Module>>(() =>
            {
                //先把当前模块添加到返回集合中
                List<Module> result = new List<Module>() { parent };
                var childrenModules = modules.FindAll(p => p.PID == parent.ID);
                if (childrenModules != null && childrenModules.Count() > 0)
                {
                    childrenModules.ForEach(module =>
                    {
                        List<Module> infos = RecursionModuleChild(module, modules);
                        if (infos != null)
                        {
                            result.AddRange(infos);
                        }
                    });
                }
                return result;
            }), "根据ID为[" + (parent == null ? string.Empty : parent.ID) + "]的模块获取所有下级模块信息异常！");
        }

        #endregion

        #region 获取检查类型列表
        /// <summary>
        /// 获取检查类型列表
        /// </summary>
        /// <returns></returns>
        public static List<CheckInfo> GetCheckInfos()
        {
            List<CheckInfo> infos = new List<CheckInfo>();
            AIDicomSliceConfigInfo configInfo = GetAIDicomSliceConfigInfo();
            if (configInfo != null)
            {
                return configInfo.SliceRuleConfigInfos.Select(x => x.StudyType)
                    .Distinct()
                    .Select(i =>
                    {
                        return new CheckInfo()
                        {
                            ID = i,
                            PID = ApplicationEx.RootID,
                            Name = i,
                            CheckCategoryType = CheckCategoryTypes.分组
                        };
                    }).ToList();
            }
            return infos;
        }

        #endregion

        #region 获取医院树结构

        private static bool ContainsAnyHospitalInRemoteConfig(RemoteExaminationConfig examConfig, HashSet<string> hospitalIdSet)
        {
            if (examConfig == null)
                return false;

            try
            {
                var config = ElincloudServer.Common.Utils.JsonUtil.FromJson<RemoteExaminationConfigInfo>(examConfig.RemoteExamination);
                if (config == null || config.HospitalRoleRelas == null || config.HospitalRoleRelas.Count == 0)
                    return false;

                return config.HospitalRoleRelas.Any(item => hospitalIdSet.Contains(item.HospitalID));
            }
            catch
            {
                return false;
            }
        }
        public static List<HospitalTreeNodeInfo> GetHospitalTreeNodeInfo(List<string> hospitalIDs = null)
        {
            List<HospitalInfo> hospitals = CommonBL.GetHospitalInfos();
            List<RemoteExaminationConfig> configs = DBServices.ElincloudConfigInstance.GetMuch<RemoteExaminationConfig>();
            var hospitalConfigs = from hospital in hospitals
                                  join config in configs
                                  on hospital.Code equals config.HospitalCode into tempConfigs
                                  from config in tempConfigs.DefaultIfEmpty()
                                  select new
                                  {
                                      Hospital = hospital,
                                      Config = config // 可能为 null
                                  };
            if (hospitalIDs != null && hospitalIDs.Any())
            {
                var idSet = new HashSet<string>(hospitalIDs);
                hospitals = hospitals.Where(h => idSet.Contains(h.ID)).ToList();
                hospitalConfigs = hospitalConfigs.Where(c => idSet.Contains(c.Hospital.ID)
                || ContainsAnyHospitalInRemoteConfig(c.Config, idSet)).ToList();
            }

            // 1. 医院字典
            var hospitalDict = hospitals.ToDictionary(h => h.ID);

            // 2. 构建有效父子关系（只保留上级存在于医院字典中的关系）
            var childrenMap = new Dictionary<string, List<string>>();
            var childWithValidParent = new HashSet<string>(); // 记录有有效上级的下级

            foreach (var cfg in hospitalConfigs)
            {
                if (cfg.Config == null)
                    continue;
                if (string.IsNullOrEmpty(cfg.Config.RemoteExamination))
                    continue;
                var config = ElincloudServer.Common.Utils.JsonUtil.FromJson<RemoteExaminationConfigInfo>(cfg.Config.RemoteExamination);
                if (config == null || config.HospitalRoleRelas.Count == 0)
                    continue;

                // 检查是否有至少一个有效上级
                bool hasValidParent = config.HospitalRoleRelas.Any(item => hospitalDict.ContainsKey(item.HospitalID));
                if (hasValidParent && hospitalDict.ContainsKey(cfg.Hospital.ID))
                {
                    childWithValidParent.Add(cfg.Hospital.ID);
                }

                // 为每个有效上级添加子节点
                foreach (var item in config.HospitalRoleRelas)
                {
                    var parentCode = item.HospitalID;
                    if (!hospitalDict.ContainsKey(parentCode))
                        continue; // 忽略无效上级

                    if (!childrenMap.ContainsKey(parentCode))
                        childrenMap[parentCode] = new List<string>();

                    // 避免重复添加相同子节点（防止配置重复）
                    if (!childrenMap[parentCode].Contains(cfg.Hospital.ID))
                        childrenMap[parentCode].Add(cfg.Hospital.ID);
                }
            }

            // 3. 确定根节点：不在 childWithValidParent 中的医院
            var rootCodes = hospitals
                .Select(h => h.ID)
                .Except(childWithValidParent)
                .ToList();

            // 若最终根列表仍为空（极少数情况），则全部作为根
            if (rootCodes.Count == 0)
                rootCodes = hospitals.Select(h => h.ID).ToList();

            // 如果没有根（全循环），将所有医院作为根
            if (rootCodes.Count == 0)
                rootCodes = hospitals.Select(h => h.ID).ToList();
            // 2. 用于记录所有已访问节点
            var visitedCodes = new HashSet<string>();

            // 4. 递归构建树（带循环检测）
            HospitalTreeNodeInfo BuildTree(string code, HashSet<string> path)
            {
                if (!hospitalDict.TryGetValue(code, out var hospital))
                    return null;
                visitedCodes.Add(code);  // 记录访问
                var node = new HospitalTreeNodeInfo { Hospital = hospital };

                if (childrenMap.TryGetValue(code, out var childCodes))
                {
                    foreach (var childCode in childCodes)
                    {
                        // 如果子节点已在当前路径中，则不添加为叶子（不继续递归）
                        if (path.Contains(childCode))
                        {
                            // 创建叶子节点（复制医院信息）
                            //if (hospitalDict.TryGetValue(childCode, out var childHospital))
                            //{
                            //    visitedCodes.Add(childCode);
                            //    node.Children.Add(new HospitalTreeNode { Hospital = childHospital });
                            //}
                        }
                        else
                        {
                            path.Add(childCode);
                            var childNode = BuildTree(childCode, path);
                            path.Remove(childCode);
                            if (childNode != null)
                                node.Children.Add(childNode);
                        }
                    }
                }
                return node;
            }

            // 5. 构建森林
            var forest = new List<HospitalTreeNodeInfo>();
            foreach (var rootCode in rootCodes)
            {
                var path = new HashSet<string> { rootCode };
                var tree = BuildTree(rootCode, path);
                if (tree != null)
                    forest.Add(tree);
            }
            var allHospitalCodes = hospitals.Select(h => h.ID).ToHashSet();
            var missingCodes = allHospitalCodes.Except(visitedCodes).ToList();
            foreach (var code in missingCodes)
            {
                var path = new HashSet<string> { code };
                var tree = BuildTree(code, path);
                if (tree != null)
                    forest.Add(tree);
            }
            return forest.OrderBy(i => i.Hospital.ID).ToList();
        }
        #endregion

        #region 资源文件路径转换  以接口形式访问

        /// <summary>
        /// 资源文件路径转换  以接口形式访问
        /// </summary>
        /// <param name="url">文件路径</param>
        /// <param name="operation">url的操作方式</param>
        /// <param name="seriesKey">唯一标识</param>
        /// <param name="isAbsolute">是否是绝对路径</param>
        /// <returns></returns>
        public static string ConvertResourceURL(string url, URLOperations operation, string seriesKey = "", bool isAbsolute = false)
        {
            if (string.IsNullOrWhiteSpace(url))
            {
                return url;
            }
            else
            {
                string ekey = ElincloudServer.Common.Utils.StringUtil.Decrypt(ElincloudServer.Common.ApplicationEx.EncryptKey);
                string keyStr = ElincloudServer.Common.Utils.StringUtil.Encrypt(url, ekey);//将实际要访问的资源文件相对路径进行加密
                string operationStr = ElincloudServer.Common.Utils.StringUtil.Encrypt(operation.ToString(), ekey); // 操作类型
                string dateStr = ElincloudServer.Common.Utils.StringUtil.Encrypt(DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"), ekey);//增加资源文件访问时效   目前是10分钟
                return string.Format(ElincloudServer.Common.ApplicationEx.ResourcesVistUrlPre, dateStr, seriesKey, keyStr, operationStr, ElincloudServer.Common.Utils.StringUtil.Encrypt(isAbsolute.ToString(), ekey));
            }
        }

        /// <summary>
        /// 获取报告图像列表
        /// </summary>
        /// <param name="relaFileName"></param>
        /// <remarks>
        /// </remarks>
        public static List<KeyValueModel> GetReportImages(string reportImageRelaFileName, bool isEncrypt = true)
        {
            List<KeyValueModel> ReportImages = new List<KeyValueModel>();
            if (!string.IsNullOrEmpty(reportImageRelaFileName))
            {
                string[] list = reportImageRelaFileName.Split('|');

                foreach (var item in list)
                {
                    string[] i = item.Split('$');
                    if (i.Count() > 0)
                    {
                        string url = i[0];
                        string description = string.Empty;
                        if (i.Count() > 1)
                        {
                            description = i[1];
                        }
                        ReportImages.Add(new KeyValueModel()
                        {
                            Key = isEncrypt == true ? ConvertResourceURL(url, URLOperations.Browse) : url,
                            Value = description,
                        });
                    }
                }
            }
            return ReportImages;
        }

        /// <summary>
        /// 获取报告图片路径
        /// </summary>
        /// <param name="describe">是否存在描述</param>
        /// <returns></returns>
        public static string GetReportImageRelaFileName(List<KeyValueModel> reportImages, bool describe, bool isEncrypt = true)
        {
            if (reportImages == null || reportImages.Count == 0)
            {
                return "";
            }
            List<string> relaPathList = new List<string>();
            foreach (var item in reportImages)
            {
                string str = item.Key;
                if (isEncrypt)
                {
                    str = GetFilePathByEncryptedUrl(str);
                }
                if (describe)
                {
                    str += '$' + item.Value;
                }
                relaPathList.Add(str);
            }
            return string.Join('|', relaPathList);
        }

        /// <summary>
        /// 根据加密url获取文件路径，并返回标准的/格式的路径
        /// </summary>
        /// <param name="url"></param>
        /// <returns></returns>
        public static string GetFilePathByEncryptedUrl(string url)
        {
            string ekey = ElincloudServer.Common.Utils.StringUtil.Decrypt(ElincloudServer.Common.ApplicationEx.EncryptKey);
            var queryParams = QueryHelpers.ParseQuery(url)["k"];
            string path = ElincloudServer.Common.Utils.StringUtil.Decrypt(queryParams, ekey);
            path = path.Replace('/', Path.DirectorySeparatorChar).Replace('\\', Path.DirectorySeparatorChar);
            return path;
        }
        #endregion
    }
}