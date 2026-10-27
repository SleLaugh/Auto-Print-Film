using System.Diagnostics;
using System.Globalization;
using System.Linq.Expressions;

// The process methods inserted below are copied at test time from the service.
// External service dependencies are replaced by these deliberately small stubs.
enum SliceImageStatus { 未切割, 正在切割, 切割失败, 已切割 }
class SlicePlanConfigInfo { public string ID = "plan"; }
class RISExamination { public string ID = "test"; public int SliceImageStatus; }
class HospitalInfo { }
namespace ElincloudServer.Common.Utils
{
    static class EnumUtil
    {
        public static T GetEnumValue<T>(int value) where T : Enum
            => (T)Enum.ToObject(typeof(T), value);
    }
}
class AIFilmSliceRecord
{
    public string ID = Guid.NewGuid().ToString();
    public string InputDir = Environment.CurrentDirectory;
    public string OutputDir = "";
    public string CacheDir = "legacy-cache";
    public string ExaminationID = "test", SeriesInstanceUID = "test";
    public string StudyTypeName = "CT", SlicePartName = "test", PlaneConfigJson = "[]";
    public string SliceConfigIDs = "plan";
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
class Config { public string ExecutablePath = ""; public int MaxConcurrentCount = 8; }
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
    private List<AIFilmSliceRecord> StudyRecords = new();
    private Action<List<AIFilmSliceRecord>>? MonitorStudy;

    // Startup restores processes before the service dispatches parallel studies.
    private void RestoreRunningProcesses() { }
    private static string BuildWindowPresetJson(IEnumerable<SlicePlanConfigInfo> plans) => "[]";
    private List<AIFilmSliceRecord> CreateMissingSeriesTasks(RISExamination examination,
        List<SlicePlanConfigInfo> plans, SqlAccess database, HospitalInfo hospital) => StudyRecords;
    private void MonitorTaskCompletion(RISExamination examination, List<AIFilmSliceRecord> records,
        List<SlicePlanConfigInfo> plans, SqlAccess database, SliceImageStatus originalStatus)
        => MonitorStudy?.Invoke(records);

    // ACTUAL_SERVICE_METHODS

    public void Dispatch(AIFilmSliceRecord record, int maximum, SqlAccess database,
        IEnumerable<AIFilmSliceRecord>? seriesRecords = null)
        => StartUnprocessedSeries(record, new() { new() }, maximum, database, seriesRecords);
    public void RestoreTrackedProcess(int processID, DateTime startTime, string outputDir)
        => TrackRunningProcess(processID, startTime, outputDir);
    public void RunStudy(List<AIFilmSliceRecord> records, SqlAccess database,
        Action<List<AIFilmSliceRecord>> monitor)
    {
        StudyRecords = records;
        MonitorStudy = monitor;
        ProcessSingleStudy(new(), new() { new() }, database, new());
    }
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
    static AIFilmSliceRecord Completed(string series, string part = "completed")
    {
        var record = Record(series, part);
        record.SliceImageStatus = (int)SliceImageStatus.已切割;
        return record;
    }
    static string CacheDirectory(string series)
        => Path.Combine(Environment.CurrentDirectory, series, "cache");
    static void SeedRawCache(string series)
    {
        string nifti = Path.Combine(CacheDirectory(series), "nifti");
        Directory.CreateDirectory(nifti);
        File.WriteAllText(Path.Combine(nifti, "raw_0000.nii.gz"), "published raw");
    }
    static void SeedSegmentationCache(string series, string model = "292", string? marker = null)
    {
        string segmentation = Path.Combine(CacheDirectory(series), model, "segmentation");
        Directory.CreateDirectory(segmentation);
        File.WriteAllText(Path.Combine(segmentation, "seg.nii.gz"), "published segmentation");
        if (marker != null) File.WriteAllText(Path.Combine(segmentation, "model.json"), marker);
    }
    static void SeedCache(string series, string? marker = null)
    {
        SeedRawCache(series);
        SeedSegmentationCache(series, marker: marker);
    }
    static void Check(bool value, string message)
    {
        if (!value) throw new Exception(message);
    }
    static SqlAccess Database(Worker worker) => new()
    {
        BeforeUpdate = () => Check(!worker.HoldsProcessLock, "Database write held admission lock"),
    };
    static void ParallelDispatch(Worker worker, AIFilmSliceRecord[] records, int maximum,
        IEnumerable<AIFilmSliceRecord>? seriesRecords = null)
    {
        using Barrier barrier = new(records.Length);
        List<Exception> errors = new();
        Thread[] threads = records.Select(record => new Thread(() =>
        {
            try
            {
                barrier.SignalAndWait();
                worker.Dispatch(record, maximum, Database(worker), seriesRecords);
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

        // A completed predecessor releases distinct parts of its series in parallel.
        using (Worker worker = new())
        {
            SeedCache("same-series", "{\"model_id\":\"292\"}");
            var records = Enumerable.Range(0, 8).Select(i => Record("same-series", $"part-{i}")).ToArray();
            var completed = Completed("same-series");
            completed.OutputDir = Path.Combine(Environment.CurrentDirectory, "same-series", ".", "completed")
                + Path.DirectorySeparatorChar;
            ParallelDispatch(worker, records, 8, records.Append(completed).ToArray());
            Check(Started(records) == 8 && worker.LiveCount == 8, "Same-series parts could not overlap");
            Check(records.Select(record => record.CacheDir).Distinct().Count() == 1,
                "Same-series tasks stopped sharing cache");
        }

        // The same normalized output path remains exclusive even with spare capacity.
        using (Worker worker = new())
        {
            SeedCache("same-output", "{\"model_id\":292}");
            var records = Enumerable.Range(0, 8).Select(i => Record("same-output")).ToArray();
            for (int i = 1; i < records.Length; i += 2)
                records[i].OutputDir = Path.Combine(Environment.CurrentDirectory, "same-output", ".", "part")
                    + Path.DirectorySeparatorChar;
            ParallelDispatch(worker, records, 8, records.Append(Completed("same-output")).ToArray());
            Check(Started(records) == 1 && worker.LiveCount == 1, "Same normalized output started twice");
        }

        // Without a reusable cache, eight callers may start only one real child.
        // Neither an unrelated success nor a failed same-series record suffices.
        using (Worker worker = new())
        {
            var records = Enumerable.Range(0, 8).Select(i => Record("cold-series", $"part-{i}")).ToArray();
            var failedPredecessor = Record("cold-series", "failed");
            failedPredecessor.SliceImageStatus = (int)SliceImageStatus.切割失败;
            var history = records.Concat(new[] { Completed("unrelated-series"), failedPredecessor }).ToArray();
            ParallelDispatch(worker, records, 8, history);
            Check(Started(records) == 1 && worker.LiveCount == 1,
                "Cold series admitted multiple children without a finished producer and reusable cache");

            var otherSeries = Record("independent-series");
            worker.Dispatch(otherSeries, 8, Database(worker), history);
            Check(Started(new[] { otherSeries }) == 1 && worker.LiveCount == 2,
                "A cold series blocked an independent series or occupied waiting slots");

            var leader = records.Single(record => record.SliceImageStatus == (int)SliceImageStatus.正在切割);
            using (Process process = Process.GetProcessById(leader.ProcessID))
            {
                process.Kill(entireProcessTree: true);
                process.WaitForExit();
            }
            leader.SliceImageStatus = (int)SliceImageStatus.切割失败;
            var waiting = records.Where(record => record.SliceImageStatus == (int)SliceImageStatus.未切割).ToArray();
            ParallelDispatch(worker, waiting, 8, history);
            Check(Started(waiting) == 1 && worker.LiveCount == 2,
                "Failed leader did not permit exactly one replacement");

            var replacement = waiting.Single(record => record.SliceImageStatus == (int)SliceImageStatus.正在切割);
            using (Process process = Process.GetProcessById(replacement.ProcessID))
            {
                process.Kill(entireProcessTree: true);
                process.WaitForExit();
            }
            replacement.SliceImageStatus = (int)SliceImageStatus.已切割;
            SeedCache("cold-series");
            var remaining = records.Where(record => record.SliceImageStatus == (int)SliceImageStatus.未切割).ToArray();
            ParallelDispatch(worker, remaining, 8, history);
            Check(Started(remaining) == remaining.Length && worker.LiveCount == remaining.Length + 1,
                "Successful replacement did not release all remaining same-series tasks");
        }

        // Startup-restored live output ownership also enforces the cold-series gate.
        using (Worker worker = new())
        {
            using Process restored = Process.Start(new ProcessStartInfo
            {
                FileName = args[0], UseShellExecute = false, CreateNoWindow = true,
            }) ?? throw new Exception("Could not start restored child");
            worker.RestoreTrackedProcess(restored.Id, restored.StartTime,
                Record("restored-series", "previous").OutputDir);
            var records = Enumerable.Range(0, 8).Select(i => Record("restored-series", $"part-{i}")).ToArray();
            ParallelDispatch(worker, records, 8, records);
            Check(Started(records) == 0 && worker.LiveCount == 1,
                "Startup-restored cold series admitted another child");

            var independent = Record("restored-independent");
            worker.Dispatch(independent, 8, Database(worker));
            Check(Started(new[] { independent }) == 1 && worker.LiveCount == 2,
                "Restored series blocked unrelated work");
        }

        // A published cache does not release followers before the first process
        // exits, even if its record was already marked as a downstream failure.
        using (Worker worker = new())
        {
            const string series = "active-with-cache";
            var leader = Record(series, "leader");
            worker.Dispatch(leader, 8, Database(worker));
            SeedCache(series);
            var followers = Enumerable.Range(0, 4).Select(i => Record(series, $"part-{i}")).ToArray();
            var history = followers.Prepend(leader).ToArray();
            ParallelDispatch(worker, followers, 8, history);
            Check(Started(followers) == 0 && worker.LiveCount == 1,
                "Published cache released followers while the first task was active");
            leader.SliceImageStatus = (int)SliceImageStatus.切割失败;
            leader.Message = "seg.nii.gz 中未找到目标Label体素: vertebrae_C1";
            ParallelDispatch(worker, followers, 8, history);
            Check(Started(followers) == 0 && worker.LiveCount == 1,
                "Failure status released followers before the real process exited");
            using (Process process = Process.GetProcessById(leader.ProcessID))
            {
                process.Kill(entireProcessTree: true);
                process.WaitForExit();
            }
            ParallelDispatch(worker, followers, 8, history);
            Check(Started(followers) == followers.Length && worker.LiveCount == followers.Length,
                "Exited no-voxel failure with published cache did not release followers");
        }

        // Completion status cannot replace actual published cache files.
        foreach (string cacheState in new[] { "absent", "raw-only", "seg-only", "temporary", "nonnumeric", "zero" })
        {
            using Worker worker = new();
            string series = "incomplete-cache-" + cacheState;
            if (cacheState is "raw-only" or "temporary" or "nonnumeric" or "zero") SeedRawCache(series);
            if (cacheState == "seg-only") SeedSegmentationCache(series);
            if (cacheState == "nonnumeric") SeedSegmentationCache(series, "unfinished-model");
            if (cacheState == "zero") SeedSegmentationCache(series, "0");
            if (cacheState == "temporary")
            {
                string work = Path.Combine(CacheDirectory(series), "292", ".segmentation-inflight");
                Directory.CreateDirectory(work);
                File.WriteAllText(Path.Combine(work, "seg.nii.gz"), "unpublished segmentation");
            }
            var predecessor = Completed(series);
            if (cacheState != "absent") predecessor.SliceImageStatus = (int)SliceImageStatus.切割失败;
            var followers = Enumerable.Range(0, 4).Select(i => Record(series, $"part-{i}")).ToArray();
            ParallelDispatch(worker, followers, 8, followers.Append(predecessor).ToArray());
            Check(Started(followers) == 1 && worker.LiveCount == 1,
                $"Incomplete cache '{cacheState}' released multiple followers");
        }

        // Model markers follow the Python legacy/string/number contract;
        // corrupt or mismatched markers must not imply a reusable cache.
        foreach (var markerCase in new (string Name, string? Marker, bool Reusable)[]
        {
            ("legacy", null, true),
            ("number", "{\"model_id\":292}", true),
            ("string", "{\"model_id\":\"292\"}", true),
            ("mismatch", "{\"model_id\":297}", false),
            ("corrupt", "{broken", false),
            ("array", "[292]", false),
            ("missing-id", "{}", false),
        })
        {
            using Worker worker = new();
            string series = "marker-" + markerCase.Name;
            SeedCache(series, markerCase.Marker);
            var failed = Record(series, "no-voxels");
            failed.SliceImageStatus = (int)SliceImageStatus.切割失败;
            var followers = Enumerable.Range(0, 4).Select(i => Record(series, $"part-{i}")).ToArray();
            ParallelDispatch(worker, followers, 8, followers.Append(failed).ToArray());
            int expected = markerCase.Reusable ? followers.Length : 1;
            Check(Started(followers) == expected && worker.LiveCount == expected,
                $"Cache marker '{markerCase.Name}' produced incorrect admission");
        }

        // One corrupt model does not hide another complete published model.
        using (Worker worker = new())
        {
            const string series = "mixed-model-cache";
            SeedCache(series, "{broken");
            SeedSegmentationCache(series, "297", "{\"model_id\":297}");
            var failed = Record(series, "failed");
            failed.SliceImageStatus = (int)SliceImageStatus.切割失败;
            var followers = Enumerable.Range(0, 4).Select(i => Record(series, $"part-{i}")).ToArray();
            ParallelDispatch(worker, followers, 8, followers.Append(failed).ToArray());
            Check(Started(followers) == followers.Length && worker.LiveCount == followers.Length,
                "Corrupt model marker hid another complete model cache");
        }

        // The real study loop releases followers in the same pass that it
        // recognizes either success or a downstream no-voxel failure with cache.
        foreach (SliceImageStatus terminal in new[] { SliceImageStatus.已切割, SliceImageStatus.切割失败 })
        {
            using Worker worker = new();
            string series = "same-pass-release-" + terminal;
            var leader = Record(series, "leader");
            worker.Dispatch(leader, 8, Database(worker));
            SeedCache(series);
            using (Process process = Process.GetProcessById(leader.ProcessID))
            {
                process.Kill(entireProcessTree: true);
                process.WaitForExit();
            }
            var followers = Enumerable.Range(0, 4)
                .Select(i => Record(series, $"follower-{i}")).ToArray();
            var records = followers.Prepend(leader).ToList();
            int monitorCalls = 0;
            worker.RunStudy(records, Database(worker), monitored =>
            {
                monitorCalls++;
                Check(leader.SliceImageStatus == (int)SliceImageStatus.正在切割,
                    "Leader reached a terminal status before completion monitoring");
                Check(Started(followers) == 1 && worker.LiveCount == 1,
                    "Study loop released multiple followers before recognizing the finished producer");
                leader.SliceImageStatus = (int)terminal;
                if (terminal == SliceImageStatus.切割失败)
                    leader.Message = "seg.nii.gz 中未找到目标Label体素: vertebrae_C1";
            });
            Check(monitorCalls == 1 && Started(followers) == followers.Length
                    && worker.LiveCount == followers.Length,
                "Study loop deferred newly released followers to the next scan");
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
