using System.Diagnostics;
using System.Globalization;
using System.Linq.Expressions;

// The process methods inserted below are copied at test time from the service.
// External service dependencies are replaced by these deliberately small stubs.
enum SliceImageStatus { 未切割, 正在切割, 切割失败 }
class SlicePlanConfigInfo { }
class AIFilmSliceRecord
{
    public string ID = Guid.NewGuid().ToString();
    public string InputDir = Environment.CurrentDirectory;
    public string OutputDir = "";
    public string CacheDir = "legacy-cache";
    public string ExaminationID = "test", SeriesInstanceUID = "test";
    public string StudyTypeName = "CT", SlicePartName = "test", PlaneConfigJson = "[]";
    public double ScanThickness = 1;
    public int SliceImageStatus, ProcessID = -1;
    public DateTime StartDate;
    public string Message = "";
    public const string _SliceImageStatus = "SliceImageStatus", _ProcessID = "ProcessID",
        _StartDate = "StartDate", _Message = "Message";
}
class SqlAccess
{
    public Action? BeforeUpdate;
    public void Update<T>(T record, Expression<Func<T, bool>> predicate, params string[] columns)
        => BeforeUpdate?.Invoke();
}
class Config { public string ExecutablePath = ""; }
class AISliceDicomServiceUtil
{
    public static readonly AISliceDicomServiceUtil Instance = new();
    public readonly Config AIFilmSliceConfigInfo = new();
    public void PrintLogAction(string message, bool error = false, object? context = null, Exception? exception = null) { }
}

class Worker : IDisposable
{
    private static readonly DateTime MinStartDate = new(1900, 1, 1);
    private readonly object ProcessNumLock = new();
    private readonly Dictionary<int, DateTime> RunningProcesses = new();
    private readonly Dictionary<string, Dictionary<int, DateTime>> RunningOutputs = new(
        OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal);

    // Startup restores processes before the service dispatches parallel studies.
    private void RestoreRunningProcesses() { }
    private static string BuildWindowPresetJson(IEnumerable<SlicePlanConfigInfo> plans) => "[]";

    // ACTUAL_SERVICE_METHODS

    public void Dispatch(AIFilmSliceRecord record, int maximum, SqlAccess database)
        => StartUnprocessedSeries(record, new() { new() }, maximum, database);
    public bool HoldsProcessLock => Monitor.IsEntered(ProcessNumLock);
    public int LiveCount { get { RefreshRunningProcesses(); return RunningProcesses.Count; } }
    public void Dispose()
    {
        foreach (int id in RunningProcesses.Keys.ToArray())
        {
            try
            {
                using Process process = Process.GetProcessById(id);
                if (!process.HasExited) process.Kill(entireProcessTree: true);
                process.WaitForExit();
            }
            catch (ArgumentException) { }
            catch (InvalidOperationException) { }
        }
        RefreshRunningProcesses();
    }
}

static class Program
{
    static AIFilmSliceRecord Record(string series, string part = "part") => new()
    {
        OutputDir = Path.Combine(Environment.CurrentDirectory, series, part),
    };
    static void Check(bool value, string message)
    {
        if (!value) throw new Exception(message);
    }
    static SqlAccess Database(Worker worker) => new()
    {
        BeforeUpdate = () => Check(!worker.HoldsProcessLock, "Database write held admission lock"),
    };
    static void ParallelDispatch(Worker worker, AIFilmSliceRecord[] records, int maximum)
    {
        using Barrier barrier = new(records.Length);
        List<Exception> errors = new();
        Thread[] threads = records.Select(record => new Thread(() =>
        {
            try
            {
                barrier.SignalAndWait();
                worker.Dispatch(record, maximum, Database(worker));
            }
            catch (Exception exception) { lock (errors) errors.Add(exception); }
        })).ToArray();
        foreach (Thread thread in threads) thread.Start();
        foreach (Thread thread in threads) thread.Join();
        if (errors.Count != 0) throw new AggregateException(errors);
    }
    static int Started(IEnumerable<AIFilmSliceRecord> records)
        => records.Count(record => record.SliceImageStatus == (int)SliceImageStatus.正在切割);

    public static void Main(string[] args)
    {
        Config config = AISliceDicomServiceUtil.Instance.AIFilmSliceConfigInfo;
        config.ExecutablePath = args[0];

        // Eight simultaneous dispatch threads must not exceed real-process capacity.
        for (int round = 0; round < 4; round++)
        {
            using Worker worker = new();
            var records = Enumerable.Range(0, 8).Select(i => Record($"cap-{round}-{i}")).ToArray();
            ParallelDispatch(worker, records, 2);
            Check(Started(records) == 2 && worker.LiveCount == 2, "Global cap was exceeded");
        }

        // Distinct parts of one series can overlap, reusing the same series cache.
        using (Worker worker = new())
        {
            var records = Enumerable.Range(0, 8).Select(i => Record("same-series", $"part-{i}")).ToArray();
            ParallelDispatch(worker, records, 8);
            Check(Started(records) == 8 && worker.LiveCount == 8, "Same-series parts could not overlap");
            Check(records.Select(record => record.CacheDir).Distinct().Count() == 1,
                "Same-series tasks stopped sharing cache");
        }

        // The same normalized output path remains exclusive even with spare capacity.
        using (Worker worker = new())
        {
            var records = Enumerable.Range(0, 8).Select(i => Record("same-output")).ToArray();
            for (int i = 1; i < records.Length; i += 2)
                records[i].OutputDir = Path.Combine(Environment.CurrentDirectory, "same-output", ".", "part")
                    + Path.DirectorySeparatorChar;
            ParallelDispatch(worker, records, 8);
            Check(Started(records) == 1 && worker.LiveCount == 1, "Same normalized output started twice");
        }

        // A failed OS spawn must leave no phantom slot or output reservation.
        using (Worker worker = new())
        {
            config.ExecutablePath = Path.Combine(Environment.CurrentDirectory, "missing-worker");
            var failed = Record("spawn-failure");
            worker.Dispatch(failed, 1, Database(worker));
            Check(failed.SliceImageStatus == (int)SliceImageStatus.切割失败 && worker.LiveCount == 0,
                "Failed spawn retained a slot");
            config.ExecutablePath = args[0];
            var retry = Record("spawn-failure");
            worker.Dispatch(retry, 1, Database(worker));
            Check(Started(new[] { retry }) == 1, "Could not retry failed spawn");
        }

        // A failed cache migration is handled before starting any child.
        using (Worker worker = new())
        {
            bool failOnce = true;
            SqlAccess database = Database(worker);
            database.BeforeUpdate = () =>
            {
                Check(!worker.HoldsProcessLock, "Database write held admission lock");
                if (failOnce) { failOnce = false; throw new Exception("Simulated cache migration failure"); }
            };
            var failed = Record("cache-migration-failure");
            worker.Dispatch(failed, 1, database);
            Check(failed.SliceImageStatus == (int)SliceImageStatus.切割失败 && worker.LiveCount == 0,
                "Cache migration failure started a child or retained a slot");
            var next = Record("cache-migration-failure");
            worker.Dispatch(next, 1, Database(worker));
            Check(Started(new[] { next }) == 1, "Cache migration failure retained an output reservation");
        }

        // Running-state database failure after Process.Start must not lose the live child.
        using (Worker worker = new())
        {
            bool failOnce = true;
            SqlAccess database = Database(worker);
            database.BeforeUpdate = () =>
            {
                Check(!worker.HoldsProcessLock, "Database write held admission lock");
                if (failOnce) { failOnce = false; throw new Exception("Simulated database failure"); }
            };
            var failed = Record("database-failure");
            failed.CacheDir = Path.Combine(Environment.CurrentDirectory, "database-failure", "cache");
            worker.Dispatch(failed, 1, database);
            var waiting = Record("database-next");
            worker.Dispatch(waiting, 1, Database(worker));
            Check(worker.LiveCount == 1 && waiting.ProcessID == -1, "Lost child after database failure");
        }

        // Exiting a tracked child frees capacity and its exact output directory.
        using (Worker worker = new())
        {
            var first = Record("exited");
            worker.Dispatch(first, 1, Database(worker));
            using (Process process = Process.GetProcessById(first.ProcessID))
            {
                process.Kill(entireProcessTree: true);
                process.WaitForExit();
            }
            var second = Record("exited");
            worker.Dispatch(second, 1, Database(worker));
            Check(Started(new[] { second }) == 1 && worker.LiveCount == 1, "Exited child retained slot/output");
        }
        Console.WriteLine("All 7 dispatch scenarios passed");
    }
}
