using System.Management;
using System.Net;
using System.Net.NetworkInformation;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using ElincloudServer.Entity;
using log4net;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using static System.Net.Mime.MediaTypeNames;

namespace ElincloudServer.Common.CommonUtil
{
    public class StringUtil
    {
        #region GUID生成算法

        /// <summary>
        /// 返回Guid用于数据库操作，特定的时间代码可以提高检索效率
        /// </summary>
        /// <returns>COMB类型 Guid 数据</returns>
        public Guid NewGuid()
        {
            byte[] guidArray = Guid.NewGuid().ToByteArray();
            DateTime dtBase = new DateTime(1900, 1, 1);
            DateTime dtNow = DateTime.Now;
            //获取用于生成byte字符串的天数与毫秒数
            TimeSpan days = new TimeSpan(dtNow.Ticks - dtBase.Ticks);
            TimeSpan msecs = new TimeSpan(dtNow.Ticks - new DateTime(dtNow.Year, dtNow.Month, dtNow.Day).Ticks);
            //转换成byte数组
            //注意SqlServer的时间计数只能精确到1/300秒
            byte[] daysArray = BitConverter.GetBytes(days.Days);
            byte[] msecsArray = BitConverter.GetBytes((long)(msecs.TotalMilliseconds / 3.333333));

            //反转字节以符合SqlServer的排序
            Array.Reverse(daysArray);
            Array.Reverse(msecsArray);

            //把字节复制到Guid中
            Array.Copy(daysArray, daysArray.Length - 2, guidArray, guidArray.Length - 6, 2);
            Array.Copy(msecsArray, msecsArray.Length - 4, guidArray, guidArray.Length - 4, 4);
            return new Guid(guidArray);
        }

        /// <summary>
        /// 从SQL Server 返回的Guid中生成时间信息
        /// </summary>
        public DateTime GetDateFromGUID(Guid id)
        {
            DateTime baseDate = new DateTime(1900, 1, 1);
            byte[] daysArray = new byte[4];
            byte[] msecsArray = new byte[4];
            byte[] guidArray = id.ToByteArray();

            // 将GUID的日期部分复制到相应的字节数组。 
            Array.Copy(guidArray, guidArray.Length - 6, daysArray, 2, 2);
            Array.Copy(guidArray, guidArray.Length - 4, msecsArray, 0, 4);

            // 将数组反转，以将它们排列成适当的顺序。 
            Array.Reverse(daysArray);
            Array.Reverse(msecsArray);

            // 将字节转换为int 
            int days = BitConverter.ToInt32(daysArray, 0);
            int msecs = BitConverter.ToInt32(msecsArray, 0);

            DateTime date = baseDate.AddDays(days);
            date = date.AddMilliseconds(msecs * 3.333333);

            return date;
        }

        #endregion

        #region 获取指定长度随机数

        public string GetRandomCode(int length)
        {
            int ret = RNGEx.Next(1, (int)Math.Pow(10, length));
            return ret.ToString().PadLeft(length, '0');
        }

        public string GetRandomCodeBySM4(string paramsKey, string earlySecurityCodeKey, int length)
        {
            StringBuilder oddChars = new StringBuilder();
            string str = SM4EncryptECB(paramsKey, earlySecurityCodeKey);
            char[] chars = str.ToCharArray();
            for (int i = 1; i <= chars.Length; i++)
            {
                int num = 0;
                if (int.TryParse(chars[i - 1].ToString(), out num) && oddChars.Length < length) // 判断索引是否为奇数
                {
                    oddChars.Append(chars[i - 1]);
                }
            }
            if (oddChars.Length < length)
            {
                for (int i = oddChars.Length; i < length; i++)
                {
                    oddChars.Append("0");
                }
            }
            return oddChars.ToString();
        }

        #endregion

        #region 获取数据库支持最小日期

        public string GetDBMinDateTime()
        {
            return "1900-01-01 00:00:00";
        }

        #endregion

        #region 获取IPV4所有网卡IP地址

        public List<string> GetLocalIPV4s()
        {
            string name = Dns.GetHostName();
            IPHostEntry me = Dns.GetHostEntry(name);
            IPAddress[] ips = me.AddressList;
            List<string> ret = new List<string>();
            if (ips != null && ips.Length > 0)
            {
                foreach (IPAddress ip in ips)
                {
                    if (ip.AddressFamily == System.Net.Sockets.AddressFamily.InterNetwork)
                    {
                        ret.Add(ip.ToString());
                    }
                }
            }
            if (ret.Count == 0)
            {
                ret.Add("127.0.0.1");
            }
            return ret;
        }

        #endregion

        #region 获取CPU ID

        public string Get_CPUID()
        {
            try
            {
                var components = new Dictionary<string, string>();
                // 并行收集不同类型的信息
                var tasks = new List<Task>
                {
                    Task.Run(() => components["CPU"] = GetProcessorInfo()),
                    Task.Run(() => components["Mainboard"] = GetMainboardInfo()),
                    Task.Run(() => components["Bios"] = GetBiosInfo())
                };

                Task.WhenAll(tasks).Wait();

                // 对组件进行排序以确保一致性
                var sortedComponents = components.OrderBy(kv => kv.Key);
                var combinedData = new StringBuilder();

                foreach (var kvp in sortedComponents)
                {
                    // 只添加非空且非未知的值
                    if (!string.IsNullOrEmpty(kvp.Value) && !kvp.Value.EndsWith("_Unknown"))
                    {
                        combinedData.Append($"{kvp.Key}={kvp.Value}|");
                    }
                }
                // 使用HMAC-SHA256进行哈希，增加盐值
                using (var hmac = new HMACSHA256(Encoding.UTF8.GetBytes(Common.Utils.StringUtil.Decrypt(ApplicationEx.EncryptKey))))
                {
                    var hashBytes = hmac.ComputeHash(Encoding.UTF8.GetBytes(combinedData.ToString()));
                    return BitConverter.ToString(hashBytes).Replace("-", "").ToUpper();
                }
            }
            catch
            {
                return string.Empty;
            }
        }

        public string Get_CPUID_OLD()
        {
            try
            {
                var sb = new StringBuilder();
                // 操作系统平台
                sb.AppendLine(RuntimeInformation.OSDescription);
                sb.AppendLine(RuntimeInformation.OSArchitecture.ToString());

                // 主机名
                sb.AppendLine(Environment.MachineName);

                // MAC 地址（取第一个活动网卡）
                var mac = NetworkInterface
        .GetAllNetworkInterfaces()
        .Where(nic =>
            nic.OperationalStatus == OperationalStatus.Up &&
            nic.NetworkInterfaceType != NetworkInterfaceType.Loopback &&
            !nic.Description.ToLower().Contains("virtual") &&
            !nic.Description.ToLower().Contains("pseudo"))
        .Select(nic => nic.GetPhysicalAddress()?.ToString())
        .Where(addr => !string.IsNullOrWhiteSpace(addr))
        .OrderBy(addr => addr) // 排序确保稳定
        .FirstOrDefault();

                if (!string.IsNullOrEmpty(mac))
                {
                    sb.AppendLine(mac);
                }

                // CPU 核心数
                sb.AppendLine(Environment.ProcessorCount.ToString());

                // 当前用户名
                // sb.AppendLine(Environment.UserName);

                // 生成哈希（可截取一部分作为唯一标识）
                using var sha256 = SHA256.Create();
                var hashBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(sb.ToString()));
                string fullId = BitConverter.ToString(hashBytes).Replace("-", "").ToLowerInvariant(); // 64位哈希
                return fullId.Substring(0, 16).ToUpper(); // 截取前 16 个字符并大写
            }
            catch
            {
                return string.Empty;
            }
        }

        #region 获取硬件信息私有方法

        /// <summary>
        /// 获取处理器信息
        /// </summary>
        private static string GetProcessorInfo()
        {
            try
            {
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    // Windows: 获取更多CPU细节
                    var cpuInfos = new List<string>();

                    using (var searcher = new ManagementObjectSearcher("SELECT * FROM Win32_Processor"))
                    {
                        foreach (ManagementObject obj in searcher.Get())
                        {
                            cpuInfos.Add($"{obj["ProcessorId"]}:{obj["NumberOfCores"]}:{obj["MaxClockSpeed"]}");
                        }
                    }

                    return string.Join("|", cpuInfos);
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                {
                    // Linux: 提取CPU特征
                    var cpuInfo = new StringBuilder();

                    if (System.IO.File.Exists("/proc/cpuinfo"))
                    {
                        var lines = System.IO.File.ReadAllLines("/proc/cpuinfo");
                        var cpuDetails = new List<string>();

                        foreach (var line in lines)
                        {
                            if (line.StartsWith("vendor_id") || line.StartsWith("model name") ||
                                line.StartsWith("cpu MHz") || line.StartsWith("cache size"))
                            {
                                cpuDetails.Add(line.Trim());
                            }
                        }

                        // 获取CPU序列号（部分ARM处理器支持）
                        if (System.IO.File.Exists("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq"))
                        {
                            var freq = System.IO.File.ReadAllText("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq").Trim();
                            cpuDetails.Add($"freq:{freq}");
                        }

                        cpuInfo.Append(string.Join(",", cpuDetails));
                    }

                    // 尝试获取CPU微码版本
                    try
                    {
                        var process = new System.Diagnostics.Process
                        {
                            StartInfo = new System.Diagnostics.ProcessStartInfo
                            {
                                FileName = "dmidecode",
                                Arguments = "-t processor",
                                RedirectStandardOutput = true,
                                UseShellExecute = false,
                                CreateNoWindow = true
                            }
                        };
                        process.Start();
                        string output = process.StandardOutput.ReadToEnd();
                        process.WaitForExit(3000);

                        if (!string.IsNullOrEmpty(output))
                        {
                            cpuInfo.Append("|dmidecode:");
                            var lines = output.Split('\n');
                            var cpuLines = lines.Where(l => l.Contains("ID:") || l.Contains("Version:")).Take(3);
                            cpuInfo.Append(string.Join(",", cpuLines));
                        }
                    }
                    catch { }

                    return cpuInfo.ToString();
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                {
                    // macOS: 获取详细CPU信息
                    var process = new System.Diagnostics.Process
                    {
                        StartInfo = new System.Diagnostics.ProcessStartInfo
                        {
                            FileName = "sysctl",
                            Arguments = "-a",
                            RedirectStandardOutput = true,
                            UseShellExecute = false,
                            CreateNoWindow = true
                        }
                    };
                    process.Start();
                    string output = process.StandardOutput.ReadToEnd();
                    process.WaitForExit();

                    var cpuDetails = new List<string>();
                    var lines = output.Split('\n');

                    foreach (var line in lines)
                    {
                        if (line.Contains("machdep.cpu.brand_string") ||
                            line.Contains("machdep.cpu.core_count") ||
                            line.Contains("hw.cpufrequency"))
                        {
                            cpuDetails.Add(line.Trim());
                        }
                    }

                    return string.Join("|", cpuDetails);
                }
            }
            catch { }

            return "CPU_Unknown";
        }

        /// <summary>
        /// 获取主板信息
        /// </summary>
        private static string GetMainboardInfo()
        {
            try
            {
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    using (var searcher = new ManagementObjectSearcher("SELECT * FROM Win32_BaseBoard"))
                    {
                        foreach (ManagementObject obj in searcher.Get())
                        {
                            return $"{obj["Manufacturer"]}:{obj["Product"]}:{obj["SerialNumber"]}:{obj["Version"]}";
                        }
                    }
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                {
                    var boardInfo = new StringBuilder();

                    // 尝试多个可能的DMI路径
                    string[] dmiPaths = {
                    "/sys/devices/virtual/dmi/id/board_serial",
                    "/sys/class/dmi/id/board_serial",
                    "/sys/firmware/dmi/entries/2-0/raw"
                };

                    foreach (var path in dmiPaths)
                    {
                        if (System.IO.File.Exists(path))
                        {
                            try
                            {
                                boardInfo.Append(System.IO.File.ReadAllText(path).Trim());
                                boardInfo.Append("|");
                            }
                            catch { }
                        }
                    }

                    // 如果DMI不可用，尝试dmidecode
                    if (boardInfo.Length == 0)
                    {
                        try
                        {
                            var process = new System.Diagnostics.Process
                            {
                                StartInfo = new System.Diagnostics.ProcessStartInfo
                                {
                                    FileName = "dmidecode",
                                    Arguments = "-s baseboard-serial-number",
                                    RedirectStandardOutput = true,
                                    UseShellExecute = false,
                                    CreateNoWindow = true
                                }
                            };
                            process.Start();
                            string serial = process.StandardOutput.ReadToEnd().Trim();
                            process.WaitForExit(3000);

                            if (!string.IsNullOrEmpty(serial) && !serial.Contains("Not Specified"))
                                boardInfo.Append(serial);
                        }
                        catch { }
                    }

                    return boardInfo.Length > 0 ? boardInfo.ToString() : "Mainboard_Unknown";
                }
            }
            catch { }

            return "Mainboard_Unknown";
        }

        /// <summary>
        /// 获取BIOS信息
        /// </summary>
        private static string GetBiosInfo()
        {
            try
            {
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    using (var searcher = new ManagementObjectSearcher("SELECT * FROM Win32_BIOS"))
                    {
                        foreach (ManagementObject obj in searcher.Get())
                        {
                            return $"{obj["Manufacturer"]}:{obj["SerialNumber"]}:{obj["Version"]}";
                        }
                    }
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                {
                    if (System.IO.File.Exists("/sys/class/dmi/id/bios_version"))
                    {
                        try
                        {
                            var biosVersion = System.IO.File.ReadAllText("/sys/class/dmi/id/bios_version").Trim();
                            var biosDate = System.IO.File.Exists("/sys/class/dmi/id/bios_date") ?
                                System.IO.File.ReadAllText("/sys/class/dmi/id/bios_date").Trim() : "";

                            return $"{biosVersion}:{biosDate}";
                        }
                        catch { }
                    }
                }
            }
            catch { }

            return "BIOS_Unknown";
        }

        #endregion

        #endregion

        #region 验证一个字符串是否是版本号

        public bool CheckVersion(string strValue)
        {
            return Regex.IsMatch(strValue, @"^\d+.\d+(\.\d+){2}$");
        }

        #endregion

        #region 格式化版本号

        public string VersionFormat(Version v, int length = 0)
        {
            string ret = "0.0.0.0";
            if (v != null)
            {
                if (length > 0)
                {
                    ret = string.Empty;
                    ret += (v.Major.ToString().Length > length ? v.Major.ToString().Substring(0, length) : v.Major.ToString()) + ".";
                    ret += (v.Minor.ToString().Length > length ? v.Minor.ToString().Substring(0, length) : v.Minor.ToString()) + ".";
                    ret += (v.Build.ToString().Length > length ? v.Build.ToString().Substring(0, length) : v.Build.ToString()) + ".";
                    ret += (v.Revision.ToString().Length > length ? v.Revision.ToString().Substring(0, length) : v.Revision.ToString());
                }
                else
                {
                    ret = v.Major + "." + v.Minor + "." + v.Build + "." + v.Revision;
                }
            }
            return ret;
        }

        #endregion

        #region GB2312加密解密

        /// <summary>
        /// GB2312编码
        /// </summary>
        /// <param name="str">待编码字符</param>
        /// <returns>编码结果</returns>
        public string EncryptToGB2312(string str)
        {
            if (!string.IsNullOrWhiteSpace(str))
            {
                StringBuilder sb = new StringBuilder();
                //GB2321的编码方式
                byte[] byStr = System.Text.Encoding.GetEncoding("GB2312").GetBytes(str);
                for (int i = 0; i < byStr.Length; i++)
                {
                    //转换为16进制方式 可选2,8,10,16进制
                    sb.Append(@"%" + Convert.ToString(byStr[i], 16));
                }
                return (sb.ToString());
            }
            return str;
        }

        /// <summary>
        /// GB2312解码
        /// </summary>
        /// <param name="str"></param>
        /// <returns></returns>
        public string DecryptFromGB2312(string str)
        {
            if (!string.IsNullOrWhiteSpace(str))
            {
                byte[] bytes = new byte[str.Split('%').Length];
                int i = 0;
                foreach (var item in str.Split('%'))
                {
                    if (item != "")
                    {
                        //转换为16进制的字节
                        bytes[i] = Convert.ToByte(item, 16);
                        i++;
                    }

                }
                //GB2312的解码
                string ret = Encoding.GetEncoding("GB2312").GetString(bytes);
                if (ret.EndsWith("\0"))
                {
                    ret = ret.Substring(0, ret.Length - 1);
                }
                return ret;
            }
            return str;
        }

        #endregion

        #region 按指定的长度截取字符串

        /// <summary>
        /// 按指定的长度截取字符串
        /// </summary>
        /// <param name="strValue"></param>
        /// <param name="maxLength"></param>
        /// <returns></returns>
        public string InterceptString(string strValue, int maxLength)
        {
            if (strValue == null)
            {
                return null;
            }
            if (strValue.Length <= maxLength)
            {
                return strValue;
            }
            return strValue.Substring(0, maxLength);
        }

        #endregion

        #region 验证一个字符串是否是 IP地址

        public bool CheckIP(string strValue)
        {

            if (strValue.Length > 16)
            {
                return false;
            }
            string[] strSplit = strValue.Split('.');
            if (strSplit.Length == 4)
            {
                bool blnChecked = CheckNumber(strSplit[0], 1, 255);
                if (blnChecked)
                {
                    for (int i = 1; i < 4; i++)
                    {
                        blnChecked = CheckNumber(strSplit[i], 0, 255);
                        if (!blnChecked)
                        {
                            return false;
                        }
                    }
                }
                else
                {
                    return false;
                }
            }
            else
            {
                return false;
            }
            return true;
        }

        /// <summary>
        /// 判断一个字符串 是否介于某个数值范围之间
        /// </summary>
        /// <param name="strValue"></param>
        /// <param name="intMin"></param>
        /// <param name="intMax"></param>
        /// <returns></returns>
        private bool CheckNumber(string strValue, int intMin, int intMax)
        {
            bool blnRtn = CheckNumericValidity(strValue);
            if (blnRtn)
            {
                int intTemp = Convert.ToInt32(strValue);
                if (intTemp <= intMax && intTemp >= intMin)
                {
                    blnRtn = true;
                }
                else
                {
                    return false;
                }
            }
            return blnRtn;
        }

        #endregion

        #region 验证一个字符串是否是 网址

        public string RemoveUrlHttpPre(string url)
        {
            if (url.StartsWith("http://"))
            {
                url = url.Replace("http://", "");
            }
            else if (url.StartsWith("https://"))
            {
                url = url.Replace("https://", "");
            }
            return url;
        }

        public bool CheckURL(string strValue)
        {
            string rex = @"^(?=^.{3,255}$)(http(s)?:\/\/)?(www\.)?[a-zA-Z0-9][-a-zA-Z0-9]{0,62}(\.[a-zA-Z0-9][-a-zA-Z0-9]{0,62})+(:\d+)*(\/[^?\s]*)?$";
            Regex reg = new Regex(rex);
            if (reg.IsMatch(strValue))
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        #endregion

        #region 获取URL中指定的参数值

        public string GetUrlParameter(string url, string paramName)
        {
            try
            {
                var uri = new Uri(url);
                var query = uri.Query;

                if (query.StartsWith('?'))
                    query = query.Substring(1); // 去除问号

                foreach (var param in query.Split('&'))
                {
                    var parts = param.Split('=');
                    if (parts.Length == 2)
                    {
                        var key = Uri.UnescapeDataString(parts[0]);
                        if (key.Equals(paramName, StringComparison.OrdinalIgnoreCase))
                        {
                            return Uri.UnescapeDataString(parts[1]); // URL 解码
                        }
                    }
                }
                return string.Empty;
            }
            catch
            {
                return string.Empty;
            }
        }

        #endregion

        #region 判断一个字符串是不是只包含数字

        /// <summary>
        /// 判断是否数字 0～9，正数或负数
        /// </summary>
        /// <param name="strValue">要判断的字符串</param>
        /// <returns>如果是数字返回 true 否则返回 false </returns>
        public bool CheckNumericValidity(string strValue)
        {
            bool blnRtn = false;
            string regx = @"^[\-]?[0-9]*[\.]?[0-9]*$";
            Regex reg = new Regex(regx);
            if (reg.IsMatch(strValue))
            {
                blnRtn = true;
            }
            return blnRtn;
        }

        /// <summary>
        /// 判断是否数字 0～9
        /// </summary>
        /// <param name="strValue">要判断的字符串</param>
        /// <returns>如果是数字返回 true 否则返回 false </returns>
        public bool CheckNumeric(string strValue)
        {
            if (strValue != string.Empty)
            {
                bool blnRtn = false;
                string regx = @"^[0-9]*$";
                Regex reg = new Regex(regx);
                if (reg.IsMatch(strValue))
                {
                    blnRtn = true;
                }
                return blnRtn;
            }
            else
            {
                return false;
            }
        }
        #endregion

        #region 检查用户输入内容中是否包含sql注入字符

        public bool CheckIsSqlInjection(string input)
        {
            ////string[] sqlCheckList = { "SELECT", "INSERT", "UPDATE", "DELETE", "DROP", "--", ";", "'" };
            ////这里把 分号 去掉主要是因为前端在传入base64图像的时候里面包含 分号
            //string[] sqlCheckList = { "SELECT", "INSERT", "UPDATE", "DELETE", "DROP", "--", "'" };
            //foreach (string item in sqlCheckList)
            //{
            //    if (input.IndexOf(item, StringComparison.OrdinalIgnoreCase) >= 0)
            //    {
            //        return true;
            //    }
            //}
            //return false;
            bool success = false;
            //string regx = @"(?i)\b(?:union|select|insert|update|delete|drop|exec|or|and)\b|--|/\*.*?\*/|'|\b(?:1=1|1' or '1'='1|0=0|0=1)\b";
            //string regx = @"(?i)\b(?:union(?:\s+all)?\s+select|select\s+.*\s+from|insert\s+into|update\s+\w+\s+set|delete\s+from|drop\s+table|exec\s+\w+|or\s+[\d'""]+\s*=\s*[\d'""]+|and\s+[\d'""]+\s*=\s*[\d'""]+)\b|--|/\*.*?\*/|(['""])\s*or\s*\1?\s*\1";
            // string regx = @"(?i)\b(?:union(?:\s+all)?\s+select|select\s+.*\s+from|insert\s+into|update\s+\w+\s+set|delete\s+from|drop\s+table|exec\s+\w+|or\s+[\d'""]+\s*=\s*[\d'""]+|and\s+[\d'""]+\s*=\s*[\d'""]+)\b|--|/\*.*?\*/|(['""])\s*or\s*\1?\s*\1|\b\d+\s*=\s*\d+\b|\b'[^']+'\s*=\s*'[^']+'\b|\b""[^""]+""\s*=\s*""[^""]+""\b";
            //2025年11月17日17:46:48 sle 注释，因为--会导致sql注入的验证不通过
            //string regx = @"""[^""]*""\s*:\s*""(?i)\b(?:union(?:\s+all)?\s+select|select\s+.*\s+from|insert\s+into|update\s+\w+\s+set|delete\s+from|drop\s+table|exec\s+\w+|or\s+\d+\s*=\s*\d+|and\s+\d+\s*=\s*\d+|or\s+true|and\s+true|or\s+['""][^'""]+['""]\s*=\s*['""][^'""]+['""]|and\s+['""][^'""]+['""]\s*=\s*['""][^'""]+['""])\b|--|/\*.*?\*/|(['""])\s*or\s*\1?\s*\1|\b'[^']+'\s*=\s*'[^']+'\b|\b""[^""]+""\s*=\s*""[^""]+""\b""";
            string regx = @"""[^""]*""\s*:\s*""(?i)\b(?:union(?:\s+all)?\s+select|select\s+.*\s+from|insert\s+into|update\s+\w+\s+set|delete\s+from|drop\s+table|exec\s+\w+|or\s+\d+\s*=\s*\d+|and\s+\d+\s*=\s*\d+|or\s+true|and\s+true|or\s+['""][^'""]+['""]\s*=\s*['""][^'""]+['""]|and\s+['""][^'""]+['""]\s*=\s*['""][^'""]+['""])\b|/\*.*?\*/|(['""])\s*or\s*\1?\s*\1|\b'[^']+'\s*=\s*'[^']+'\b|\b""[^""]+""\s*=\s*""[^""]+""\b""";

            Regex reg = new Regex(regx, RegexOptions.IgnoreCase);
            if (reg.IsMatch(input))
            {
                success = true;
            }
            return success;
        }

        #endregion

        #region 检查是否有非法字符

        /// <summary>
        /// 检查是否存在非法字符(暂定~`*/\&%#+=-!@$()|:;"'<>,.。?)
        /// </summary>
        /// <param name="Value"></param>
        /// <returns>如果没有非法字符返回 true 否则返回false</returns>
        public bool CheckStringValidity(string Value)
        {
            //上海院的登陆名全都是域用户，登陆名必须有@,所以去掉了@的判断
            string[] unlaString = { "$", "%", "#", "&", "*", "+", "<", ">", "?", "'", "=", "-", "/", "\\", "\"", ")", "(", "~", "!" };
            //20080617 wlk 调整为和windows命名的非法字符一致
            // string[] unlaString = { "<", ">", "?", "/", "\"", ":", ";", "*", ",", "=", "\\", "`", "|" };
            int intLen = unlaString.Length;
            for (int i = 0; i < intLen; i++)
            {
                int reInt = Value.IndexOf(unlaString[i]);
                if (reInt != -1)
                {
                    return false;
                }
            }
            return true;
        }

        public string RemoveInvalidationChar(string strValue)
        {
            string strRtn = strValue;
            string[] unlaString = { "<", ">", "?", "/", "\"", "：", "*", "\\", "|" };
            int intLen = unlaString.Length;
            for (int i = 0; i < intLen; i++)
            {
                strRtn = strRtn.Replace(unlaString[i], string.Empty);
            }
            return strRtn;
        }

        #endregion

        #region  检查是否是有效的电子邮件地址

        /// <summary>
        /// 检查给定的字符串是否是有效的电子邮件地址
        /// </summary>
        /// <param name="Value"></param>
        /// <returns></returns>
        public bool CheckEMailValidity(string Value)
        {
            return Regex.IsMatch(Value, @"^([\w-\.]+)@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.)|(([\w-]+\.)+))([a-zA-Z]{2,4}|[0-9]{1,3})(\]?)$");
        }

        #endregion

        #region  检查是否是有效的身份证号

        /// <summary>
        /// 检查给定的字符串是否是有效的身份证号
        /// </summary>
        /// <param name="Value"></param>
        /// <returns></returns>
        public bool CheckIDCardValidity(string Value)
        {
            int intCid = Value.Length;
            if (intCid != 15)
            {
                //double dbiSum = 0;
                //System.Text.RegularExpressions.Regex rg = new System.Text.RegularExpressions.Regex(@"^\d{17}(\d|x)$");
                //System.Text.RegularExpressions.Match mc = rg.Match(Value);
                //if (!mc.Success)
                //{
                //    return false;
                //}
                //Value = Value.ToLower();
                //Value = Value.Replace("x", "a");

                //try
                //{
                //    DateTime.Parse(Value.Substring(6, 4) + "-" + Value.Substring(10, 2) + "-" + Value.Substring(12, 2));
                //}
                //catch
                //{
                //    return false;
                //}
                //for (int i = 17; i >= 0; i--)
                //{
                //    dbiSum += (System.Math.Pow(2, i) % 11) * int.Parse(Value[17 - i].ToString().Trim(), System.Globalization.NumberStyles.HexNumber);

                //}
                //if (dbiSum % 11 != 1)
                //{
                //    return false;
                //}

                //return true;
                if (Value == null || Value.Length != 18)
                {
                    return false;
                }
                // 2. 正则校验格式：前17位数字 + 最后一位数字或X/x
                var reg = new System.Text.RegularExpressions.Regex(@"^\d{17}[\dXx]$");
                if (!reg.IsMatch(Value))
                {
                    return false;
                }
                // 统一转大写
                Value = Value.ToUpper();

                // 3. 校验出生日期是否合法
                try
                {
                    string birthDateStr = Value.Substring(6, 8);
                    DateTime birthDate = DateTime.ParseExact(birthDateStr, "yyyyMMdd", System.Globalization.CultureInfo.InvariantCulture);

                    // 出生日期不能是未来时间
                    if (birthDate > DateTime.Now)
                    {
                        return false;
                    }
                }
                catch
                {
                    return false;
                }

                // 4. 国标校验码计算（GB 11643-1999）
                int[] weights = { 7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2 };
                char[] checkCodes = { '1', '0', 'X', '9', '8', '7', '6', '5', '4', '3', '2' };

                int sum = 0;
                for (int i = 0; i < 17; i++)
                {
                    sum += (Value[i] - '0') * weights[i];
                }
                // 5. 对比最后一位是否匹配
                char lastChar = Value[17];
                return lastChar == checkCodes[sum % 11];
            }
            else
            {
                string ereg;
                System.Text.RegularExpressions.Regex rg = new System.Text.RegularExpressions.Regex(@"^\d{15}");
                System.Text.RegularExpressions.Match mc = rg.Match(Value);
                if (!mc.Success)
                {
                    return false;
                }
                else
                {

                    if ((Convert.ToInt32(Value.Substring(6, 2)) + 1900) % 4 == 0 || ((Convert.ToInt32(Value.Substring(6, 2)) + 1900) % 100 == 0 && (Convert.ToInt32(Value.Substring(6, 2)) + 1900) % 4 == 0))
                    {
                        ereg = @"^[1-9][0-9]{5}[0-9]{2}((01|03|05|07|08|10|12)(0[1-9]|[1-2][0-9]|3[0-1])|(04|06|09|11)(0[1-9]|[1-2][0-9]|30)|02(0[1-9]|[1-2][0-9]))[0-9]{3}$";//测试出生日期的合法性 
                    }
                    else
                    {
                        ereg = @"^[1-9][0-9]{5}[0-9]{2}((01|03|05|07|08|10|12)(0[1-9]|[1-2][0-9]|3[0-1])|(04|06|09|11)(0[1-9]|[1-2][0-9]|30)|02(0[1-9]|1[0-9]|2[0-8]))[0-9]{3}$";//测试出生日期的合法性 
                    }
                    if (Regex.IsMatch(Value, ereg))
                    {
                        return true;
                    }
                    else
                    {
                        return false;
                    }
                }
            }
        }

        #endregion

        #region 检查是否有效的手机号码

        /// <summary>
        /// 检查给定的字符串是否是有效的手机号码
        /// </summary>
        /// <param name="Value"></param>
        /// <returns></returns>
        public bool CheckMobileValidity(string Value)
        {
            string strMobi = Value;
            System.Text.RegularExpressions.Regex rgg = new System.Text.RegularExpressions.Regex(@"^\d{11}$");
            System.Text.RegularExpressions.Match mcc = rgg.Match(Value);
            if (!mcc.Success)
            {
                return false;
            }
            if (strMobi.Substring(0, 1) != "1")
            {
                return false;
            }
            else
            {
                return true;
            }
        }
        #endregion

        #region 检查是否有效的邮箱

        /// <summary>
        /// 检查给定的字符串是否是有效的邮箱
        /// </summary>
        /// <param name="Value"></param>
        /// <returns></returns>
        public bool CheckEmailValidity(string Value)
        {
            return Regex.IsMatch(Value, @"^([\w-\.]+)@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.)|(([\w-]+\.)+))([a-zA-Z]{2,4}|[0-9]{1,3})(\]?)$");
        }

        #endregion

        #region 是否为有效办公电话

        /// <summary>
        /// 检查给定的字符是否这有效的电话号码
        /// </summary>
        /// <param name="strValue"></param>
        /// <returns></returns>
        public bool CheckOfficePhoneValidity(string strValue)
        {
            //如：010-88486488,010-88486488-88(888 ,8888)
            //  0731-8848648,0512-88486488,0731-8848648-88(888,8888)
            //8612-12456678-1234
            string strPhone = strValue;
            Regex rgg = new Regex(@"(^0\d{2,3}-\d{7,8}$)|(^0\d{2,3}-\d{7,8}-(\d{2,3}|\d{3,4})$)|(^\d{4}-\d{7,8}-\d{3,4}$)");
            Match mcc = rgg.Match(strValue);
            if (!mcc.Success)
            {
                return false;
            }
            else
            {
                return true;
            }
        }

        #endregion

        #region 检查字符串的长度

        /// <summary>
        /// 验证一个字符串的长度和给定长度的关系
        /// </summary>
        /// <param name="Value"></param>
        /// <param name="Length"></param>
        /// <returns>小于等于给定长度 返回false,大于给定长度返回 true</returns>
        public bool CheckStringLength(string Value, int Length)
        {
            int intCount = GetStringLength(Value);
            if (intCount > Length)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        /// <summary>
        /// 检查给定的字符串是否在某个长度范围之间
        /// </summary>
        /// <param name="Value"></param>
        /// <param name="MinLength">最小长度</param>
        /// <param name="MaxLength">最大长度</param>
        /// <returns>在给定长度之间（包括给定的长度）返回 true 否则返回false</returns>
        public bool CheckStringLength(string Value, int MinLength, int MaxLength)
        {
            int intCount = GetStringLength(Value);
            if (intCount > MaxLength || intCount > MinLength)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        /// <summary>
        /// 获取字符串所占用的字节数
        /// </summary>
        /// <param name="str">输入的字符串</param>
        /// <returns></returns>
        public int GetStringLength(string str)
        {
            return Encoding.Default.GetByteCount(str);
        }

        /// <summary>
        /// 检查当前字符串中是否有中文字符
        /// </summary>
        /// <param name="strValue"></param>
        /// <returns></returns>
        public bool ContainCHS(string strValue)
        {
            bool blnRtn = false;

            for (int i = 0; i < strValue.Length; i++)
            {
                int intRtn = GetStringLength(strValue[i].ToString().Trim());
                if (intRtn > 1)
                {
                    blnRtn = true;
                }
            }
            return blnRtn;
        }

        #endregion

        #region 文件大小的转换

        /// <summary>
        /// 转文件大小转换成KB和MB
        /// </summary>
        /// <param name="lSize"></param>
        /// <returns></returns>
        public string ConvertSizeFromLong(string strSize)
        {
            if (strSize.Length == 0)
            {
                return "";
            }
            string strRtn = string.Empty;
            string strFlag = string.Empty;
            double dSize = double.Parse(strSize);
            double dRemainSize = 0;
            int intTemp = (int)dSize;
            if (intTemp != 0)
            {
                strFlag = "B";
                dRemainSize = dSize;
                intTemp = (int)(dSize / 1024);
                if (intTemp != 0)
                {
                    strFlag = "K";
                    dRemainSize = dSize / 1024;
                    intTemp = (int)(dSize / (1024 * 1024));
                    if (intTemp != 0)
                    {
                        strFlag = "M";
                        dRemainSize = dSize / (1024 * 1024);
                        intTemp = (int)(dSize / (1024 * 1024 * 1024));
                        if (intTemp != 0)
                        {
                            strFlag = "G";
                            dRemainSize = dSize / (1024 * 1024 * 1024);
                        }
                    }
                }
            }

            dRemainSize = Math.Round(dRemainSize, 2);
            strRtn = dRemainSize.ToString().Trim();
            if (strFlag != "B" && strRtn.IndexOf(".") == -1)
            {
                strRtn += ".00";
            }

            switch (strFlag)
            {
                case "B":
                    strRtn = strRtn + " 字节";
                    break;
                case "K":
                    strRtn = strRtn + " KB";
                    break;
                case "M":
                    strRtn = strRtn + " MB";
                    break;
                case "G":
                    strRtn = strRtn + " GB";
                    break;
                default:
                    break;
            }

            return strRtn;
        }

        #endregion

        #region 字符串加密解密

        public string Sha256Encrypt(string data)
        {
            if (!string.IsNullOrWhiteSpace(data))
            {
                byte[] buffer = Encoding.UTF8.GetBytes(data);
                using var sha256 = SHA256.Create();
                byte[] hashBytes = sha256.ComputeHash(buffer);
                return BitConverter.ToString(hashBytes).Replace("-", "").ToLower();
            }
            return data;
        }

        public string HashEncrypt(string data)
        {
            if (!string.IsNullOrWhiteSpace(data))
            {
                // 转换串到一个统一的字符编码标准字节数组中。
                byte[] msgBytes = Encoding.Unicode.GetBytes(data);

                // 计算一个哈希的数据。
                SHA1Managed sha1 = new SHA1Managed();
                byte[] hashBytes = sha1.ComputeHash(msgBytes);

                // 用字符串生成器增加速度。
                StringBuilder hashRet = new StringBuilder("");

                // 循环转换每一个数据在哈希表中
                foreach (byte b in hashBytes)
                    hashRet.AppendFormat("{0:X}", b);

                // 返回哈希字串
                return hashRet.ToString();
            }
            return data;
        }

        public string MD5Encrypt(string strInput)
        {
            if (!string.IsNullOrWhiteSpace(strInput))
            {
                // Create a new instance of the MD5CryptoServiceProvider object.
                MD5 md5Hasher = MD5.Create();

                // Convert the input string to a byte array and compute the hash.
                byte[] btData = md5Hasher.ComputeHash(Encoding.Default.GetBytes(strInput));

                // Create a new Stringbuilder to collect the bytes
                // and create a string.
                StringBuilder sBuilder = new StringBuilder();

                // Loop through each byte of the hashed data 
                // and format each one as a hexadecimal string.
                for (int i = 0; i < btData.Length; i++)
                {
                    sBuilder.Append(btData[i].ToString("X2"));
                }

                // Return the hexadecimal string.
                return sBuilder.ToString();
            }
            return strInput;
        }

        /// <summary>
        /// 对字符串进行加密
        /// </summary>
        /// <param name="strInput">输入的字符串</param>
        /// <returns>输出加过密的字符串</returns>	
        public string Encrypt(string strInput)
        {
            try
            {
                if (strInput != string.Empty)
                {
                    MemoryStream ms = new MemoryStream();
                    Byte[] bytearrayinput = (new UTF8Encoding()).GetBytes(strInput);

                    //具有随机密钥的 DES 实例
                    DESCryptoServiceProvider des = new DESCryptoServiceProvider();

                    //从此实例创建 DES 加密器
                    ICryptoTransform desencrypt = des.CreateEncryptor(
                        new byte[] { 8, 7, 6, 9, 4, 3, 2, 1 }
                        , new byte[] { 1, 2, 3, 4, 9, 6, 7, 8, 1, 2, 3, 4, 9, 6, 7, 8, 1, 2, 3, 4, 9, 6, 7, 8, 1, 2, 3, 4, 9, 6, 7, 8 });

                    //创建使用 des 加密转换文件流的加密流
                    CryptoStream cryptostream = new CryptoStream(ms, desencrypt, CryptoStreamMode.Write);

                    //写出 DES 加密文件
                    cryptostream.Write(bytearrayinput, 0, bytearrayinput.Length);
                    cryptostream.FlushFinalBlock();
                    cryptostream.Close();
                    ms.Close();

                    return System.Convert.ToBase64String(ms.ToArray());
                }
                else
                {
                    return strInput;
                }
            }
            catch
            {
                return strInput;
            }
        }

        /// <summary>
        /// 对字符串进行解密
        /// </summary>
        /// <param name="strInput">输入的加密字符串</param>
        /// <returns>输出解密的字符串</returns>
        public string Decrypt(string strInput)
        {
            try
            {
                if (strInput != string.Empty)
                {
                    Byte[] bytearrayinput = System.Convert.FromBase64String(strInput);
                    MemoryStream ms = new MemoryStream();

                    //从此 des 实例创建 DES 解密器
                    DESCryptoServiceProvider des1 = new DESCryptoServiceProvider();
                    ICryptoTransform desdecrypt = des1.CreateDecryptor(new byte[] { 8, 7, 6, 9, 4, 3, 2, 1 }
                    , new byte[] { 1, 2, 3, 4, 9, 6, 7, 8, 1, 2, 3, 4, 9, 6, 7, 8, 1, 2, 3, 4, 9, 6, 7, 8, 1, 2, 3, 4, 9, 6, 7, 8 });

                    //创建加密流集合以便对传入的字节进行读取并执行 des 解密转换
                    CryptoStream cryptostreamDecr = new CryptoStream(ms, desdecrypt, CryptoStreamMode.Write);
                    cryptostreamDecr.Write(bytearrayinput, 0, bytearrayinput.Length);
                    cryptostreamDecr.FlushFinalBlock();
                    cryptostreamDecr.Close();
                    ms.Close();
                    //输出已解密文件的内容
                    return (new UTF8Encoding()).GetString(ms.ToArray());
                }
                else
                {
                    return strInput;
                }
            }
            catch
            {
                return strInput;
            }
        }

        /// <summary> 
        /// 加密数据 需要密码
        /// </summary> 
        /// <param name="Text"></param> 
        /// <param name="sKey"></param> 
        /// <returns></returns> 
        public string Encrypt(string Text, string sKey)
        {
            try
            {
                if (!string.IsNullOrWhiteSpace(Text))
                {
                    DESCryptoServiceProvider des = new DESCryptoServiceProvider();
                    byte[] inputByteArray;
                    inputByteArray = Encoding.Default.GetBytes(Text);
                    string md5value = "";
                    using (var md5 = MD5.Create())
                    {
                        var result = md5.ComputeHash(Encoding.UTF8.GetBytes(sKey));
                        var strResult = BitConverter.ToString(result);
                        md5value = strResult.Replace("-", "").ToUpper().Substring(0, 8);
                    }
                    des.Key = ASCIIEncoding.ASCII.GetBytes(md5value);
                    des.IV = ASCIIEncoding.ASCII.GetBytes(md5value);
                    System.IO.MemoryStream ms = new System.IO.MemoryStream();
                    CryptoStream cs = new CryptoStream(ms, des.CreateEncryptor(), CryptoStreamMode.Write);
                    cs.Write(inputByteArray, 0, inputByteArray.Length);
                    cs.FlushFinalBlock();
                    StringBuilder ret = new StringBuilder();
                    foreach (byte b in ms.ToArray())
                    {
                        ret.AppendFormat("{0:X2}", b);
                    }
                    return ret.ToString();
                }
                return Text;
            }
            catch (Exception ex)
            {
                return string.Empty;
            }

        }

        /// <summary> 
        /// 解密数据 需要密码
        /// </summary> 
        /// <param name="Text"></param> 
        /// <param name="sKey"></param> 
        /// <returns></returns> 
        public string Decrypt(string text, string sKey)
        {
            try
            {
                if ((text.Length % 2) == 0)
                {
                    DESCryptoServiceProvider des = new DESCryptoServiceProvider();
                    int len;
                    len = text.Length / 2;
                    byte[] inputByteArray = new byte[len];
                    int x, i;
                    for (x = 0; x < len; x++)
                    {
                        i = Convert.ToInt32(text.Substring(x * 2, 2), 16);
                        inputByteArray[x] = (byte)i;
                    }
                    string md5value = "";
                    using (var md5 = MD5.Create())
                    {
                        var result = md5.ComputeHash(Encoding.UTF8.GetBytes(sKey));
                        var strResult = BitConverter.ToString(result);
                        md5value = strResult.Replace("-", "").ToUpper().Substring(0, 8);
                    }
                    des.Key = ASCIIEncoding.ASCII.GetBytes(md5value);
                    des.IV = ASCIIEncoding.ASCII.GetBytes(md5value);
                    System.IO.MemoryStream ms = new System.IO.MemoryStream();
                    CryptoStream cs = new CryptoStream(ms, des.CreateDecryptor(), CryptoStreamMode.Write);
                    cs.Write(inputByteArray, 0, inputByteArray.Length);
                    cs.FlushFinalBlock();
                    return Encoding.Default.GetString(ms.ToArray());
                }
                else
                {
                    return string.Empty;
                }
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("解密数据失败！", ex);
                return string.Empty;
            }

        }


        /// <summary>
        /// SM4 ECB加密
        /// </summary>
        /// <param name="entity"></param>
        /// <returns></returns>
        public string SM4EncryptECB(string text, string sKey)
        {
            if (!string.IsNullOrWhiteSpace(text))
            {
                Sm4Context ctx = new Sm4Context
                {
                    IsPadding = true
                };

                byte[] keyBytes = Encoding.Default.GetBytes(sKey);

                SM4 sm4 = new SM4();
                sm4.SetKeyEnc(ctx, keyBytes);
                byte[] encrypted = sm4.Sm4CryptEcb(ctx, Encoding.Default.GetBytes(text));
                StringBuilder ret = new StringBuilder();
                foreach (byte b in encrypted.ToArray())
                {
                    ret.AppendFormat("{0:X2}", b);
                }
                return ret.ToString();
            }
            return text;
        }

        /// <summary>
        /// SM4  ECB解密
        /// </summary>
        /// <param name="entity"></param>
        /// <returns></returns>
        public string SM4DecryptECB(string text, string sKey)
        {
            if (!string.IsNullOrWhiteSpace(text))
            {
                Sm4Context ctx = new Sm4Context
                {
                    IsPadding = true,
                    Mode = 0
                };

                int len;
                len = text.Length / 2;
                byte[] inputByteArray = new byte[len];
                int x, i;
                for (x = 0; x < len; x++)
                {
                    i = Convert.ToInt32(text.Substring(x * 2, 2), 16);
                    inputByteArray[x] = (byte)i;
                }

                byte[] keyBytes = Encoding.Default.GetBytes(sKey);

                SM4 sm4 = new SM4();
                sm4.Sm4SetKeyDec(ctx, keyBytes);
                byte[] decrypted = sm4.Sm4CryptEcb(ctx, inputByteArray);
                return Encoding.Default.GetString(decrypted);
            }
            return text;
        }

        public string SM4EncryptECBToBase64(string text, string sKey)
        {
            if (!string.IsNullOrWhiteSpace(text))
            {
                Sm4Context ctx = new Sm4Context
                {
                    IsPadding = true
                };

                byte[] keyBytes = Encoding.Default.GetBytes(sKey);

                SM4 sm4 = new SM4();
                sm4.SetKeyEnc(ctx, keyBytes);
                byte[] encrypted = sm4.Sm4CryptEcb(ctx, Encoding.Default.GetBytes(text));
                return Convert.ToBase64String(encrypted);
            }
            return text;
        }

        /// <summary>
        /// 对字符串进行加密
        /// </summary>
        /// <param name="strInput">输入的字符串</param>
        /// <returns>输出加过密的字符串</returns>	
        public string EncryptToHex(string strInput)
        {
            try
            {
                if (strInput != string.Empty)
                {
                    MemoryStream ms = new MemoryStream();
                    Byte[] bytearrayinput = (new UTF8Encoding()).GetBytes(strInput);

                    //具有随机密钥的 AES 实例
                    AesCryptoServiceProvider aes = new AesCryptoServiceProvider();

                    //从此实例创建 DES 加密器
                    ICryptoTransform desencrypt = aes.CreateEncryptor(
                        new byte[] { 8, 7, 6, 9, 4, 3, 2, 1, 8, 7, 6, 9, 4, 3, 2, 1 }
                        , new byte[] { 1, 2, 3, 4, 9, 6, 7, 8, 1, 2, 3, 4, 9, 6, 7, 8 });

                    //创建使用 des 加密转换文件流的加密流
                    CryptoStream cryptostream = new CryptoStream(ms, desencrypt, CryptoStreamMode.Write);

                    //写出 DES 加密文件
                    cryptostream.Write(bytearrayinput, 0, bytearrayinput.Length);
                    cryptostream.FlushFinalBlock();
                    cryptostream.Close();
                    ms.Close();
                    StringBuilder ret = new StringBuilder();
                    foreach (byte b in ms.ToArray())
                    {
                        ret.AppendFormat("{0:X2}", b);
                    }
                    return ret.ToString();
                }
                else
                {
                    return strInput;
                }
            }
            catch
            {
                return string.Empty;
            }
        }

        /// <summary>
        /// 对字符串进行解密
        /// </summary>
        /// <param name="strInput">输入的加密字符串</param>
        /// <returns>输出解密的字符串</returns>
        public string DecryptToHex(string strInput)
        {
            try
            {
                if (strInput != string.Empty)
                {
                    //Byte[] bytearrayinput = System.Convert.FromBase64String(strInput);
                    MemoryStream ms = new MemoryStream();
                    int len;
                    len = strInput.Length / 2;
                    byte[] bytearrayinput = new byte[len];
                    int x, i;
                    for (x = 0; x < len; x++)
                    {
                        i = Convert.ToInt32(strInput.Substring(x * 2, 2), 16);
                        bytearrayinput[x] = (byte)i;
                    }

                    //具有随机密钥的 AES 实例
                    AesCryptoServiceProvider aes = new AesCryptoServiceProvider();
                    ICryptoTransform desdecrypt = aes.CreateDecryptor(new byte[] { 8, 7, 6, 9, 4, 3, 2, 1, 8, 7, 6, 9, 4, 3, 2, 1 }
                    , new byte[] { 1, 2, 3, 4, 9, 6, 7, 8, 1, 2, 3, 4, 9, 6, 7, 8 });

                    //创建加密流集合以便对传入的字节进行读取并执行 des 解密转换
                    CryptoStream cryptostreamDecr = new CryptoStream(ms, desdecrypt, CryptoStreamMode.Write);
                    cryptostreamDecr.Write(bytearrayinput, 0, bytearrayinput.Length);
                    cryptostreamDecr.FlushFinalBlock();
                    cryptostreamDecr.Close();
                    ms.Close();
                    //输出已解密文件的内容
                    return (new UTF8Encoding()).GetString(ms.ToArray());
                }
                else
                {
                    return strInput;
                }
            }
            catch
            {
                return string.Empty;
            }
        }

        #endregion

        #region 检查是否是有效的文件夹或文件名称

        /// <summary>
        /// 检查文件夹名称的合法性，合法则为　true,　否则为　false
        /// </summary>
        /// <param name="strValue"></param>
        /// <returns></returns>
        public bool CheckDirectoryNameValidity(string strValue)
        {
            if (strValue == null || strValue.Trim() == string.Empty)
            {
                return false;//zhangyue
            }

            if (strValue.StartsWith(".") || strValue.EndsWith("."))
            {
                return false;
            }

            char[] charString = new char[11] { '\\', '/', ':', '*', '?', '%', '"', '<', '>', '|', '\'' };//zhangyue
            for (int i = 0; i < charString.Length; i++)
            {
                if (strValue.IndexOf(charString[i]) != -1)
                {
                    return false;
                }
            }

            return true;

        }

        /// <summary>
        /// 检查文件名的合法性，合法则为　true,　否则为　false
        /// </summary>
        /// <param name="strValue"></param>
        /// <returns></returns>
        public bool CheckFileNameValidity(string strValue)
        {
            if (strValue == null || strValue.Trim() == string.Empty)
            {
                return false;//zhangyue
            }
            //if (strValue.StartsWith("."))
            //{
            //    return false;
            //}
            char[] charString = new char[10] { '\\', '/', ':', '*', '?', '"', '<', '>', '|', '\'' };//zhangyue
            for (int i = 0; i < charString.Length; i++)
            {
                if (strValue.IndexOf(charString[i]) != -1)
                {
                    return false;
                }
            }
            return true;
        }

        /// <summary>
        /// 文件名称是否含有特殊符号----zhangyue
        /// </summary>
        /// <param name="strValue"></param>
        /// <returns></returns>
        public bool IsFileNameHasSpecSign(string strValue)
        {
            if (strValue == null || strValue.Trim() == string.Empty)
            {
                return false;
            }

            char[] charString = new char[11] { '\\', '/', ':', '*', '?', '%', '"', '<', '>', '|', '\'' };
            for (int i = 0; i < charString.Length; i++)
            {
                if (strValue.IndexOf(charString[i]) != -1)
                {
                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// 判断扩展名是否有特殊符号
        /// </summary>
        /// <param name="strValue"></param>
        /// <returns></returns>
        public bool IsExtHasSpecSign(string strValue)
        {
            if (strValue == null || strValue.Trim() == string.Empty)
            {
                return false;
            }

            char[] charString = new char[11] { '\\', '/', ':', '*', '?', '%', '"', '<', '>', '|', '\'' };
            for (int i = 0; i < charString.Length; i++)
            {
                if (strValue.IndexOf(charString[i]) != -1)
                {
                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="strValue"></param>
        /// <returns></returns>
        public bool CheckPathValidity(string strValue)
        {
            Regex re = new Regex(@"^(([a-zA-Z]:\\)|(\\{2}\w+)\$?)((([^/\\\?\*])(\\?))*)$");
            return re.IsMatch(strValue);
        }
        #endregion

        #region 字符串是否为Int32整数

        /// <summary>
        /// 字符串是否为Int32整数
        /// </summary>
        /// <param name="Value"></param>
        /// <returns></returns>
        public bool CheckStringInt32(string Value)
        {
            try
            {
                Convert.ToInt32(Value);
            }
            catch
            {
                return false;
            }
            return true;
        }
        #endregion

        #region 通配符比较，支持"*"，"?"
        public bool CompareWildcard(string WildString, string Mask, bool IgnoreCase)
        {
            int i = 0, k = 0;
            while (k != WildString.Length)
            {
                if (i >= Mask.Length) return false;
                switch (Mask[i])
                {
                    case '*':
                        if ((i + 1) == Mask.Length)
                            return true;
                        while (k != WildString.Length)
                        {
                            if (CompareWildcard(WildString.Substring(k), Mask.Substring(i + 1), IgnoreCase))
                                return true;
                            k += 1;
                        }
                        return false;

                    case '?':
                        break;

                    default:
                        if (IgnoreCase == false && WildString[k] != Mask[i])
                            return false;
                        if (IgnoreCase && Char.ToLower(WildString[k]) != Char.ToLower(Mask[i]))
                            return false;
                        break;
                }
                i += 1;
                k += 1;
            }
            if (k == WildString.Length)
            {
                if (i == Mask.Length || Mask.Substring(i).Replace("*", "") == "")
                    return true;
            }
            return false;
        }

        public bool CompareWildcard(string WildString, List<string> Masks, bool IgnoreCase)
        {
            for (int i = 0; i < Masks.Count; ++i)
            {
                if (CompareWildcard(WildString, Masks[i], IgnoreCase))
                {
                    return true;
                }
            }
            return false;
        }
        #endregion

        #region 检查是否有效的车牌号

        /// <summary>
        /// 检查给定的字符串是否是有效的车牌号
        /// </summary>
        /// <param name="Value"></param>
        /// <returns></returns>
        public bool CheckCarNumberValidity(string Value)
        {
            string strCarNo = Value;
            //System.Text.RegularExpressions.Regex reg = new System.Text.RegularExpressions.Regex(@"^[\u4E00-\u9FA5][\da-zA-Z]{6}$");"^[/u4e00-/u9fa5]{1}[A-Z0-9]{6}$";
            System.Text.RegularExpressions.Regex reg = new System.Text.RegularExpressions.Regex(@"^[\u4E00-\u9FA5]{1}[A-Z]{1}[A-Z0-9]{5}$");
            if (reg.IsMatch(Value))
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        #endregion

        #region 获取汉字的拼音简码
        public string GetSpellCode(string text)
        {
            StringBuilder myStr = new StringBuilder();
            char[] txtChars = text.ToCharArray();
            foreach (char c in txtChars)
            {
                byte[] arrCN = Encoding.Default.GetBytes(c.ToString());
                if (arrCN.Length > 1)
                {
                    int area = (short)arrCN[0];
                    int pos = (short)arrCN[1];
                    int code = (area << 8) + pos;
                    int[] areacode = { 45217, 45253, 45761, 46318, 46826, 47010, 47297, 47614, 48119, 48119, 49062, 49324, 49896, 50371, 50614, 50622, 50906, 51387, 51446, 52218, 52698, 52698, 52698, 52980, 53689, 54481 };
                    for (int j = 0; j < 26; j++)
                    {
                        int max = 55290;
                        if (j != 25) max = areacode[j + 1];
                        if (areacode[j] <= code && code < max)
                        {
                            myStr.Append(Encoding.Default.GetString(new byte[] { (byte)(65 + j) }));
                        }
                    }
                }
                else
                {
                    myStr.Append(c);
                }
            }
            return myStr.ToString();
        }
        #endregion

        #region 重写 System.IO.Path 相关方法
        public string Combine(params string[] paths)
        {
            if (paths.Length == 0)
            {
                throw new ArgumentException("please input path");
            }
            else
            {
                StringBuilder builder = new StringBuilder();
                string spliter = ElincloudServer.Common.Utils.ConfigUtil.IsWindow ? "\\" : "/";
                string firstPath = ElincloudServer.Common.Utils.ConfigUtil.IsWindow ? paths[0].Replace("/", spliter) : paths[0].Replace("\\", spliter);

                if (firstPath.StartsWith("HTTP", StringComparison.OrdinalIgnoreCase))
                {
                    spliter = "/";
                }

                if (!firstPath.EndsWith(spliter))
                {
                    firstPath = firstPath + spliter;
                }
                builder.Append(firstPath);

                for (int i = 1; i < paths.Length; i++)
                {
                    string nextPath = ElincloudServer.Common.Utils.ConfigUtil.IsWindow ? paths[i].Replace("/", spliter) : paths[i].Replace("\\", spliter);
                    if (nextPath != null)
                    {
                        if (nextPath.StartsWith("/") || nextPath.StartsWith("\\"))
                        {
                            nextPath = nextPath.Substring(1);
                        }

                        if (i != paths.Length - 1)//not the last one
                        {
                            if (nextPath.EndsWith("/") || nextPath.EndsWith("\\"))
                            {
                                nextPath = nextPath.Substring(0, nextPath.Length - 1) + spliter;
                            }
                            else
                            {
                                nextPath = nextPath + spliter;
                            }
                        }
                        builder.Append(nextPath);
                    }
                }
                return builder.ToString();
            }
        }

        /// <summary>
        /// 重写 Utils.StringUtil.GetFileName 方法
        /// </summary>
        /// <param name="path"></param>
        /// <returns></returns>
        /// <remarks>
        /// sle 2025年9月4日11:04:03
        /// 因为在系统类型和文件路径的分隔符不一致时会导致解析出错，比如linux解析带\\分隔符的路径时，会认为\\是字符串而不是分隔符
        /// </remarks>
        public string GetFileName(string path)
        {
            if (string.IsNullOrEmpty(path))
            {
                return path;
            }
            // 统一替换为当前系统分隔符
            string normalizedPath = path
                .Replace('\\', Path.DirectorySeparatorChar)
                .Replace('/', Path.DirectorySeparatorChar);

            return Path.GetFileName(normalizedPath);
        }

        /// <summary>
        /// 重写 Utils.StringUtil.GetFileNameWithoutExtension 方法
        /// </summary>
        /// <param name="path"></param>
        /// <returns></returns>
        /// <remarks>
        /// sle 2025年9月4日11:04:03
        /// 因为在系统类型和文件路径的分隔符不一致时会导致解析出错，比如linux解析带\\分隔符的路径时，会认为\\是字符串而不是分隔符
        /// </remarks>
        public string GetFileNameWithoutExtension(string path)
        {
            if (string.IsNullOrEmpty(path))
            {
                return path;
            }

            string normalizedPath = path
                .Replace('\\', Path.DirectorySeparatorChar)
                .Replace('/', Path.DirectorySeparatorChar);

            return Path.GetFileNameWithoutExtension(normalizedPath);
        }

        /// <summary>
        /// 重写 Utils.StringUtil.GetDirectoryName 方法
        /// </summary>
        /// <param name="path"></param>
        /// <returns></returns>
        /// <remarks>
        /// sle 2025年9月4日11:04:03
        /// 因为在系统类型和文件路径的分隔符不一致时会导致解析出错，比如linux解析带\\分隔符的路径时，会认为\\是字符串而不是分隔符
        /// </remarks>
        public string GetDirectoryName(string path)
        {
            if (string.IsNullOrEmpty(path))
                return path;

            string normalizedPath = path
                .Replace('\\', Path.DirectorySeparatorChar)
                .Replace('/', Path.DirectorySeparatorChar);

            return Path.GetDirectoryName(normalizedPath);
        }
        #endregion

        #region 字符集检测

        public bool IsTextUTF8(byte[] inputStream)
        {
            int encodingBytesCount = 0;
            bool allTextsAreASCIIChars = true;

            for (int i = 0; i < inputStream.Length; i++)
            {
                byte current = inputStream[i];

                if ((current & 0x80) == 0x80)
                {
                    allTextsAreASCIIChars = false;
                }
                // First byte
                if (encodingBytesCount == 0)
                {
                    if ((current & 0x80) == 0)
                    {
                        // ASCII chars, from 0x00-0x7F
                        continue;
                    }

                    if ((current & 0xC0) == 0xC0)
                    {
                        encodingBytesCount = 1;
                        current <<= 2;

                        // More than two bytes used to encoding a unicode char.
                        // Calculate the real length.
                        while ((current & 0x80) == 0x80)
                        {
                            current <<= 1;
                            encodingBytesCount++;
                        }
                    }
                    else
                    {
                        // Invalid bits structure for UTF8 encoding rule.
                        return false;
                    }
                }
                else
                {
                    // Following bytes, must start with 10.
                    if ((current & 0xC0) == 0x80)
                    {
                        encodingBytesCount--;
                    }
                    else
                    {
                        // Invalid bits structure for UTF8 encoding rule.
                        return false;
                    }
                }
            }

            if (encodingBytesCount != 0)
            {
                // Invalid bits structure for UTF8 encoding rule.
                // Wrong following bytes count.
                return false;
            }

            // Although UTF8 supports encoding for ASCII chars, we regard as a input stream, whose contents are all ASCII as default encoding.
            return !allTextsAreASCIIChars;
        }

        public bool IsTextGB18030(byte[] bytes)
        {
            // GB18030特征：双字节或四字节序列
            for (int i = 0; i < bytes.Length; i++)
            {
                if (bytes[i] >= 0x81 && bytes[i] <= 0xFE)
                {
                    // 双字节序列
                    if (i + 1 < bytes.Length &&
                        ((bytes[i + 1] >= 0x40 && bytes[i + 1] <= 0x7E) ||
                         (bytes[i + 1] >= 0x80 && bytes[i + 1] <= 0xFE)))
                    {
                        i++; // 跳过第二个字节
                        continue;
                    }
                    // 四字节序列
                    else if (i + 3 < bytes.Length &&
                             bytes[i + 1] >= 0x30 && bytes[i + 1] <= 0x39 &&
                             bytes[i + 2] >= 0x81 && bytes[i + 2] <= 0xFE &&
                             bytes[i + 3] >= 0x30 && bytes[i + 3] <= 0x39)
                    {
                        i += 3;
                        continue;
                    }
                    return false;
                }
            }
            return true;
        }

        #endregion

        #region DicomTagValidate

        public string ValidateDicomTag_AE(string content)
        {
            // may not be longer than 16 characters
            if (content.Length > 16)
            {
                //throw new Dicom.DicomValidationException(content, Dicom.DicomVR.AE, "value exceeds maximum length of 16 characters");
                return "AE长度不能超过16个字符！";
            }
            // may not contain only of spaces
            if (System.Text.RegularExpressions.Regex.IsMatch(content, @"^\s*$"))
            {
                //throw new Dicom.DicomValidationException(content, Dicom.DicomVR.AE, "value may not consist only of spaces");
                return "AE不能为空！";
            }
            // Default Character Repertoire excluding character code 5CH (the BACKSLASH "\" in ISO-IR 6), and control characters LF, FF, CR and ESC.
            if (content.Contains("\\") || content.ToCharArray().Any(Char.IsControl))
            {
                //throw new Dicom.DicomValidationException(content, Dicom.DicomVR.AE, "value contains invalid control character");
                return "AE中存在非法字符！";
            }
            return string.Empty;
        }

        #endregion

        #region 获取DicomUID 个别国产PACS厂家的UID中有非法字符

        public string GetDicomUID(string inputString)
        {
            string ret = string.Empty;
            // 定义要匹配的模式
            Regex regex = new Regex("[0-9.]+");
            Match match = regex.Match(inputString);
            if (match.Success)
            {
                ret = match.Value;
            }
            return ret;
        }

        #endregion

        #region 获取Dicom Modality 个别国产PACS厂家的Modality中有中文

        public string GetDicomModality(string inputString)
        {
            string ret = string.Empty;
            // 定义要匹配的模式
            Regex regex = new Regex("[A-Z0-9_ ]+");
            Match match = regex.Match(inputString);
            if (match.Success)
            {
                ret = match.Value;
            }
            return ret;
        }

        #endregion

        #region 生成一个指定长度的随机复杂密码

        public string GeneratePwd(int len)
        {
            string chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*()-_=+<>?:;";
            Random randrom = new Random((int)DateTime.Now.Ticks);
            string str = "";
            for (int j = 0; j < 10000; j++)
            {
                for (int i = 0; i < len; i++)
                {
                    str += chars[randrom.Next(chars.Length)];//randrom.Next(int i)返回一个小于所指定最大值的非负随机数
                }
                if (str.Trim("0123456789".ToCharArray()) == "")//判断是否全是数字
                {
                    str = string.Empty;
                    continue;
                }
                else if (str.Trim("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz".ToCharArray()) == "")//判断是否全是字母
                {
                    str = string.Empty;
                    continue;
                }
                else
                {
                    break;
                }
            }
            return str;
        }

        #endregion

        #region 验证登录账号不允许存在英文、数字以及下划线之外的字符

        public bool LoginNameValid(string input)
        {
            string pattern = "^[a-zA-Z0-9_]+$";
            Regex regex = new Regex(pattern);
            return regex.IsMatch(input);
        }

        #endregion

        #region 判断字符串是否为json对象

        public bool IsJsonString(string jsonString)
        {
            try
            {
                JToken token = JToken.Parse(jsonString);
                return token.Type == JTokenType.Object || token.Type == JTokenType.Array;
            }
            catch (JsonReaderException)
            {
                // JSON 字符串无效
                return false;
            }
        }

        #endregion

        #region 移除字符串中的标点符号

        public string RemovePunctuations(string value)
        {
            List<string> unlaString = new List<string>() { " ", "。", ".", "？", "?", "！", "!", "，", ",", "、", "；", ";", "：", ":", "”", "\"", "“", "（", "(", "）", ")", "【", "[", "】", "]", "{", "}", "──", "《", "》", "<", ">", "·", "—", "*", "×", "□", "/", "\\", "▲", "●", "～", "→" };
            string ret = "";
            for (int i = 0; i < value.Length; i++)
            {
                string subStr = value.Substring(i, 1);
                if (!unlaString.Contains(subStr))
                {
                    ret += subStr;
                }
            }
            return ret;
        }

        #endregion

        #region 年龄转换

        public string ConvertChineseAgeToBirthday(DateTime dtBirthday, int YearValue = 5)
        {
            DateTime dtNow = DateTime.Now;
            string strAge = string.Empty; // 年龄的字符串表示

            int intYear = dtNow.Year - dtBirthday.Year;

            int intMonth = dtNow.Month - dtBirthday.Month;
            if (intMonth < 0)
            {
                intMonth += 12;
                intYear -= 1;
            }

            int intDay = dtNow.Day - dtBirthday.Day;
            if (intDay < 0)
            {
                intDay += DateTime.DaysInMonth(dtNow.Year, dtNow.Month);
                intMonth -= 1;
            }

            int intHour = dtNow.Hour - dtBirthday.Hour;
            if (intHour < 0)
            {
                intHour += 24;
                intDay -= 1;
            }

            bool isYear = intYear > 0;
            bool isMonth = intMonth > 0;
            bool isDay = intDay > 0;
            bool isHours = intHour > 0;

            if (isYear)
            {
                strAge = string.Format("{0}岁", intYear);
                if (isMonth && YearValue > intYear)
                {
                    strAge = string.Format("{0}{1}月", strAge, intMonth);
                }
                return strAge;
            }

            if (isMonth)
            {
                strAge = string.Format("{0}月", intMonth);
                if (isDay)
                {
                    strAge = string.Format("{0}{1}天", strAge, intDay);
                }
                return strAge;
            }

            if (isDay)
            {
                strAge = string.Format("{0}天", intDay);
                if (isHours)
                {
                    strAge = string.Format("{0}{1}小时", strAge, intHour);
                }
                return strAge;
            }

            if (isHours)
            {
                strAge = string.Format("{0}小时", intHour);
                return strAge;
            }
            return strAge;
        }

        public string ConvertDicomAgeToChineseAge(string dicomAge)
        {
            string patientAge = dicomAge;
            if (dicomAge.Contains("Y"))
            {
                string ageStr = dicomAge.Replace("Y", "");
                int ageInt = 0;
                if (int.TryParse(ageStr, out ageInt))
                {
                    patientAge = ageInt.ToString() + "岁";
                }
            }
            else if (dicomAge.Contains("M"))
            {
                string ageStr = dicomAge.Replace("M", "");
                int ageInt = 0;
                if (int.TryParse(ageStr, out ageInt))
                {
                    patientAge = ageInt.ToString() + "月";
                }
            }
            else if (dicomAge.Contains("W"))
            {
                string ageStr = dicomAge.Replace("W", "");
                int ageInt = 0;
                if (int.TryParse(ageStr, out ageInt))
                {
                    patientAge = ageInt.ToString() + "周";
                }
            }
            else if (dicomAge.Contains("D"))
            {
                string ageStr = dicomAge.Replace("D", "");
                int ageInt = 0;
                if (int.TryParse(ageStr, out ageInt))
                {
                    patientAge = ageInt.ToString() + "天";
                }
            }
            return patientAge;
        }

        public string ConvertChineseAgeToDicomAge(string chineseAge)
        {
            string patientAge = "";

            if (chineseAge.Contains("岁"))
            {
                patientAge = chineseAge.Substring(0, chineseAge.IndexOf("岁") + 1).Replace("岁", "Y").PadLeft(4, '0');
            }
            else if (chineseAge.Contains("月"))
            {
                patientAge = chineseAge.Substring(0, chineseAge.IndexOf("月") + 1).Replace("月", "M").PadLeft(4, '0');
            }
            else if (chineseAge.Contains("天"))
            {
                patientAge = chineseAge.Substring(0, chineseAge.IndexOf("天") + 1).Replace("天", "D").PadLeft(4, '0');
            }
            else
            {
                patientAge = "000Y";
            }

            return patientAge;
        }

        public DateTime ConvertDicomAgeToBirthday(string dicomAge, DateTime? checkDate)
        {
            DateTime birthday = DateTime.Now;
            if (checkDate != null)
            {
                birthday = Convert.ToDateTime(checkDate);
            }
            if (dicomAge.Contains("Y"))
            {
                string ageStr = dicomAge.Replace("Y", "");
                int ageInt = 0;
                if (int.TryParse(ageStr, out ageInt))
                {
                    birthday = birthday.AddYears(-ageInt);
                }
            }
            else if (dicomAge.Contains("M"))
            {
                string ageStr = dicomAge.Replace("M", "");
                int ageInt = 0;
                if (int.TryParse(ageStr, out ageInt))
                {
                    birthday = birthday.AddMonths(-ageInt);
                }
            }
            else if (dicomAge.Contains("W"))
            {
                string ageStr = dicomAge.Replace("W", "");
                int ageInt = 0;
                if (int.TryParse(ageStr, out ageInt))
                {
                    ageInt = ageInt * 7;
                    birthday = birthday.AddDays(-ageInt);
                }
            }
            else if (dicomAge.Contains("D"))
            {
                string ageStr = dicomAge.Replace("D", "");
                int ageInt = 0;
                if (int.TryParse(ageStr, out ageInt))
                {
                    birthday = birthday.AddDays(-ageInt);
                }
            }
            return birthday;
        }

        #endregion

        #region 性别转换

        public Sexs ConvertDicomSexToEnumSex(string dicomSex)
        {
            if (dicomSex.ToLower() == "m")
            {
                return Sexs.男;
            }
            else if (dicomSex.ToLower() == "f")
            {
                return Sexs.女;
            }
            else
            {
                return Sexs.未知;
            }
        }

        #endregion

        #region 时间转换
        /// <summary>
        /// 分钟转小时
        /// </summary>
        /// <param name="minutes"></param>
        /// <returns></returns>
        public string ConvertToHoursAndMinutes(double minutes)
        {
            int hours = (int)minutes / 60;
            int remainingMinutes = (int)minutes % 60;
            if (hours > 0)
            {
                string result = string.Format("{0}小时", hours);
                if (remainingMinutes > 0)
                {
                    result = string.Format("{0}{1}分钟", result, remainingMinutes);
                }
                return result;
            }
            else
            {
                return string.Format("{0}分钟", remainingMinutes);
            }
        }

        #endregion

        #region  验证字符串中是否包含请求的IP
        /// <summary>
        /// 判断字符串中是否包含请求的IP
        /// </summary>
        /// <param name="requestip"></param>
        /// <param name="sourceStr"></param>
        /// <returns></returns>
        public bool IsContainsRequestIP(string requestip, string sourceStr)
        {
            if (requestip == "127.0.0.1")
            {
                return true;
            }
            if (!string.IsNullOrEmpty(sourceStr))
            {
                if (sourceStr.Contains(requestip))
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }
            else
            {
                return false;
            }
        }

        #endregion
    }
}
