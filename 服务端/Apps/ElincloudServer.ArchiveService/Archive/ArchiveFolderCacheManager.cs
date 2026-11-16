using System.Timers;

/// <summary>
/// 缓存实体
/// </summary>
public class ArchiveFolderCache
{
    private string patientID = string.Empty;
    public string PatientID
    {
        get { return patientID; }
        set { patientID = value; }
    }

    private string archiveFolderName = string.Empty;
    public string ArchiveFolderName
    {
        get { return archiveFolderName; }
        set { archiveFolderName = value; }
    }

    private DateTime cacheDate = DateTime.Now;
    public DateTime CacheDate
    {
        get { return cacheDate; }
        set { cacheDate = value; }
    }
}

/// <summary>
/// 内存缓存 + TXT持久化，2分钟定时清理过期(1天)并落盘
/// </summary>
public class ArchiveFolderCacheManager
{
    // 缓存文件路径
    private readonly string _cacheFilePath;
    // 内存缓存集合
    private readonly List<ArchiveFolderCache> _memoryList = new List<ArchiveFolderCache>();
    private readonly object _lockObj = new object();
    private readonly System.Timers.Timer _workTimer;

    // 配置参数
    private const int IntervalMs = 2 * 60 * 1000;    // 2分钟执行一次
    private const int ExpireDay = 1;                 // 过期天数1天

    public ArchiveFolderCacheManager(string cacheFilePath)
    {
        _cacheFilePath = cacheFilePath;

        // 启动加载文件
        LoadFromFile();

        // 启动定时器
        _workTimer = new System.Timers.Timer(IntervalMs);
        _workTimer.Elapsed += TimerExecute;
        _workTimer.AutoReset = true;
        _workTimer.Start();
    }

    /// <summary>
    /// 添加一条缓存（仅写入内存）
    /// CacheDate默认当前时间，外部也可自行指定
    /// </summary>
    public void Add(ArchiveFolderCache item)
    {
        if (item == null) return;
        lock (_lockObj)
        {
            // 可选：如需去重，根据PatientID判断
            // _memoryList.RemoveAll(x => x.PatientID == item.PatientID);
            _memoryList.Add(item);
        }
    }

    /// <summary>
    /// 根据PatientID查询未过期记录
    /// </summary>
    public ArchiveFolderCache GetByPatientId(string patientId)
    {
        lock (_lockObj)
        {
            var expireTime = DateTime.Now.AddDays(-ExpireDay);
            return _memoryList.FirstOrDefault(x =>
                x.PatientID == patientId && x.CacheDate > expireTime);
        }
    }

    /// <summary>
    /// 定时器执行：清理过期 + 写入文件
    /// </summary>
    private void TimerExecute(object sender, ElapsedEventArgs e)
    {
        ClearExpiredData();
        SaveToFile();
    }

    /// <summary>
    /// 清理超过1天的数据
    /// </summary>
    public void ClearExpiredData()
    {
        lock (_lockObj)
        {
            DateTime expireTime = DateTime.Now.AddDays(-ExpireDay);
            _memoryList.RemoveAll(item => item.CacheDate <= expireTime);
        }
    }

    /// <summary>
    /// 从txt加载数据到内存
    /// 格式：PatientID|ArchiveFolderName|CacheDate
    /// </summary>
    public void LoadFromFile()
    {
        lock (_lockObj)
        {
            if (!File.Exists(_cacheFilePath))
                return;
            try
            {
                var lines = File.ReadAllLines(_cacheFilePath);
                _memoryList.Clear();
                foreach (var line in lines)
                {
                    var parts = line.Split('|');
                    if (parts.Length != 3) continue;

                    string pid = parts[0];
                    string folder = parts[1];
                    if (DateTime.TryParse(parts[2], out var cacheDt))
                    {
                        var model = new ArchiveFolderCache
                        {
                            PatientID = pid,
                            ArchiveFolderName = folder,
                            CacheDate = cacheDt
                        };
                        _memoryList.Add(model);
                    }
                }
                // 加载完成立刻清理过期项
                ClearExpiredData();
            }
            catch (Exception ex)
            {
                // 日志记录
                Console.WriteLine($"加载缓存文件异常：{ex.Message}");
            }
        }
    }

    /// <summary>
    /// 将内存数据写入txt
    /// </summary>
    private void SaveToFile()
    {
        lock (_lockObj)
        {
            try
            {
                List<string> lines = new List<string>();
                foreach (var item in _memoryList)
                {
                    // 使用o标准时间格式，稳定序列化解析
                    lines.Add($"{item.PatientID}|{item.ArchiveFolderName}|{item.CacheDate:o}");
                }
                File.WriteAllLines(_cacheFilePath, lines);
                Console.WriteLine($"[{DateTime.Now}] 缓存写入完成，当前有效记录：{_memoryList.Count}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"写入缓存文件异常：{ex.Message}");
            }
        }
    }
}