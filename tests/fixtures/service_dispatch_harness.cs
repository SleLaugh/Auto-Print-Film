using System.Diagnostics;
using System.Globalization;
using System.Linq.Expressions;

// The process methods inserted below are copied at test time from the service.
// External service dependencies are replaced by these deliberately small stubs.
enum SliceImageStatus { 未切割, 正在切割, 切割失败, 已切割, 不切割 }
enum DicomImageCategories { 原始影像 }
class SlicePlanConfigInfo
{
    public string ID = "plan", StudyTypeName = "CT";
    public bool IsEnabel = true;
}
class RISExamination
{
    public string ID = "test", PatientID = "patient", ImageExaminationID = "image", ImagePatientID = "image-patient";
    public int SliceImageStatus;
}
class HospitalInfo { public string ID = "hospital", Name = "hospital", HospitalGroupID = "hospital", Code = "hospital"; }
class ExaminationImage
{
    public string ExaminationID = "image", PatientID = "image-patient", SeriesInstanceUID = "test";
    public int SliceImageStatus, Category;
    public bool ReceiveCompleted = true;
}
class V_ExaminationImage
{
    public string ExaminationID = "test", PatientID = "patient", ImageExaminationID = "image", ImagePatientID = "image-patient";
    public string StudyTypeName = "CT", SeriesInstanceUID = "test";
    public DateTime StudyDate = DateTime.MinValue;
    public int SliceImageStatus;
    public bool IsUrgent;
}
class Preset { public List<object> BodyPartMappings = new() { new() }; }
class AIDicomSliceContext
{
    public HospitalInfo Hospital = new();
    public SqlAccess Instance = new();
    public List<SlicePlanConfigInfo> SlicePlans = new() { new() };
    public Preset Preset = new();
}
static class DBServices
{
    public static readonly Dictionary<string, SqlAccess> Databases = new();
    public static SqlAccess GetHospitalDBInstance(string id) => Databases[id];
}
static class CommonBL
{
    public static List<SlicePlanConfigInfo> GetSlicePlanConfigInfos(string id) => new() { new() };
    public static Preset GetAIDicomSlicePresetConfigInfo(string code) => new();
}
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
    public Action<object, string[]>? ObserveUpdate;
    public Action<Type>? BeforeRead;
    public List<V_ExaminationImage> PendingSeries = new();
    public readonly Dictionary<string, List<AIFilmSliceRecord>> StudyRecords = new();
    public List<T> GetMuch<T>(Expression<Func<T, bool>>? predicate = null)
    {
        BeforeRead?.Invoke(typeof(T));
        IEnumerable<T> items = typeof(T) == typeof(V_ExaminationImage)
            ? PendingSeries.Cast<T>() : typeof(T) == typeof(AIFilmSliceRecord)
            ? StudyRecords.Values.SelectMany(records => records).Cast<T>() : Enumerable.Empty<T>();
        return predicate == null ? items.ToList() : items.Where(predicate.Compile()).ToList();
    }
    public void Update<T>(T record, Expression<Func<T, bool>> predicate, params string[] columns)
    {
        ObserveUpdate?.Invoke(record!, columns);
        BeforeUpdate?.Invoke();
    }
}
class Config { public string ExecutablePath = ""; public int MaxConcurrentCount = 8; }
class AISliceDicomServiceUtil
{
    public static readonly AISliceDicomServiceUtil Instance = new();
    public readonly Config AIFilmSliceConfigInfo = new();
    public List<HospitalInfo> HospitalInfos = new();
    public void PrintLogAction(string message, bool error = false, object? context = null, Exception? exception = null) { }
}

class Worker : IDisposable
{
    private static readonly DateTime MinStartDate = new(1900, 1, 1);
    private readonly object ProcessNumLock = new();
    private readonly Dictionary<int, DateTime> RunningProcesses = new();
    private readonly Dictionary<string, Dictionary<int, DateTime>> RunningOutputs = new(
        OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal);
    private string? NextHospitalID;
    private bool RunningProcessesRestored;
    private Action<List<AIFilmSliceRecord>>? MonitorStudy;
    public Action<List<AIFilmSliceRecord>, AIDicomSliceContext>? MonitorHospital;

    // ACTUAL_HOSPITAL_QUEUE

    private static string BuildWindowPresetJson(IEnumerable<SlicePlanConfigInfo> plans, AIDicomSliceContext context) => "[]";
    private List<AIFilmSliceRecord> CreateMissingSeriesTasks(RISExamination examination,
        List<V_ExaminationImage> series, AIDicomSliceContext context) => context.Instance.StudyRecords[examination.ID];
    private void MonitorPendingSeriesTasks(List<AIFilmSliceRecord> records, AIDicomSliceContext context)
    {
        MonitorStudy?.Invoke(records);
        MonitorHospital?.Invoke(records, context);
    }
    private static void SetExaminationStatus(RISExamination examination, SliceImageStatus status,
        AIDicomSliceContext context) => examination.SliceImageStatus = (int)status;
    private static void UpdateExaminationImageStatus(V_ExaminationImage series, SliceImageStatus status,
        AIDicomSliceContext context) => series.SliceImageStatus = (int)status;

    // ACTUAL_SERVICE_METHODS

    public bool Dispatch(AIFilmSliceRecord record, int maximum, SqlAccess database,
        IEnumerable<AIFilmSliceRecord>? seriesRecords = null)
        => StartUnprocessedSeries(record, new() { new() }, maximum, new() { Instance = database }, seriesRecords);
    public Process LaunchForArguments(string cacheDir, string outputDir)
        => StartAutoPrintFilm(8, Environment.CurrentDirectory, cacheDir, outputDir,
            "CT", "肺部", 1.5, "[]", "[]", out _)
            ?? throw new Exception("Argument regression child was not admitted");
    public void RestoreTrackedProcess(int processID, DateTime startTime, string outputDir)
        => TrackRunningProcess(processID, startTime, outputDir);
    public void RunStudy(List<AIFilmSliceRecord> records, SqlAccess database,
        Action<List<AIFilmSliceRecord>> monitor)
    {
        database.StudyRecords["test"] = records;
        MonitorStudy = monitor;
        AIDicomSliceContext context = new() { Instance = database };
        var prepared = ProcessPendingSeries(new(), new() { new() }, context);
        DispatchHospitals(8, (context, prepared));
    }
    public void RunBusiness() => BusinessLogic();
    public void DispatchHospitals(int maximum,
        params (AIDicomSliceContext Context, List<AIFilmSliceRecord> Records)[] hospitals)
    {
        AISliceDicomServiceUtil.Instance.AIFilmSliceConfigInfo.MaxConcurrentCount = maximum;
        DispatchHospitalTasks(hospitals.Select(hospital => new HospitalDispatchQueue
        {
            Context = hospital.Context,
            Tasks = new(hospital.Records.Select(record => (record, hospital.Records))),
        }).ToList());
    }
    public bool HoldsProcessLock => Monitor.IsEntered(ProcessNumLock);
    public int LiveCount { get { RefreshRunningProcesses(); return RunningProcesses.Count; } }
    public void ReleaseProcesses()
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
    public void Dispose() => ReleaseProcesses();
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

    static (AIDicomSliceContext Context, List<AIFilmSliceRecord> Records) Hospital(
        Worker worker, string id, int taskCount, List<string> starts)
    {
        var hospital = new HospitalInfo { ID = id, Name = id, HospitalGroupID = id, Code = id };
        SqlAccess database = Database(worker);
        database.ObserveUpdate = (record, columns) =>
        {
            if (record is AIFilmSliceRecord task && task.ProcessID > 0
                && task.SliceImageStatus == (int)SliceImageStatus.正在切割
                && columns.Contains(AIFilmSliceRecord._ProcessID)) starts.Add(id);
        };
        return (new() { Hospital = hospital, Instance = database },
            Enumerable.Range(0, taskCount).Select(i => Record($"{id}-series-{i}")).ToList());
    }

    static void AddStudy(AIDicomSliceContext context, AIFilmSliceRecord record, string id,
        bool urgent = false, DateTime? studyDate = null)
    {
        record.ExaminationID = id;
        record.SeriesInstanceUID = id + "-series";
        context.Instance.StudyRecords[id] = new() { record };
        context.Instance.PendingSeries.Add(new()
        {
            ExaminationID = id, SeriesInstanceUID = record.SeriesInstanceUID,
            IsUrgent = urgent, StudyDate = studyDate ?? DateTime.Today,
        });
    }

    static void TestHospitalFairness(string executable)
    {
        Config config = AISliceDicomServiceUtil.Instance.AIFilmSliceConfigInfo;

        TestStudyDateOrdering();

        // With five free slots, each hospital starts once before anyone starts twice.
        using (Worker worker = new())
        {
            List<string> starts = new();
            var a = Hospital(worker, "first-A", 5, starts);
            var b = Hospital(worker, "first-B", 5, starts);
            var c = Hospital(worker, "first-C", 5, starts);
            worker.DispatchHospitals(5, a, b, c);
            Check(starts.SequenceEqual(new[] { "first-A", "first-B", "first-C", "first-A", "first-B" })
                && worker.LiveCount == 5, "Hospitals did not share the first round of slots fairly");
        }

        // The scan order is always A/B/C, while a single released slot must rotate
        // across scans. Full-capacity scans must never advance that stored priority.
        using (Worker worker = new())
        {
            List<string> starts = new();
            var hospitals = new[] { "cycle-A", "cycle-B", "cycle-C" }
                .Select(id => Hospital(worker, id, 4, starts)).ToArray();
            for (int round = 0; round < 9; round++)
            {
                worker.DispatchHospitals(1, hospitals);
                Check(starts.Count == round + 1 && starts[^1] == hospitals[round % 3].Context.Hospital.ID,
                    "A busy first hospital starved a later hospital across scans");
                for (int fullScan = 0; fullScan <= round % 3; fullScan++)
                    worker.DispatchHospitals(1, hospitals);
                Check(starts.Count == round + 1 && worker.LiveCount == 1,
                    "A full-capacity scan admitted another child");
                worker.ReleaseProcesses();
                foreach (var hospital in hospitals)
                    foreach (var record in hospital.Records.Where(record => record.ProcessID > 0))
                        record.SliceImageStatus = (int)SliceImageStatus.已切割;
            }
        }

        // Sharing is work-conserving: an active hospital can consume all spare slots.
        using (Worker worker = new())
        {
            List<string> starts = new();
            var active = Hospital(worker, "single-active", 7, starts);
            worker.DispatchHospitals(4, Hospital(worker, "empty-before", 0, starts), active,
                Hospital(worker, "empty-after", 0, starts));
            Check(Started(active.Records) == 4 && worker.LiveCount == 4,
                "Empty hospitals prevented the only active hospital from filling capacity");
        }

        // Skip empty, unconfigured, and cold-series blocked candidates, including a
        // blocked head within an otherwise runnable hospital's own queue.
        using (Worker worker = new())
        {
            List<string> starts = new();
            var blocked = Hospital(worker, "blocked", 2, starts);
            var leader = Record("blocked-cold", "leader");
            worker.Dispatch(leader, 4, Database(worker));
            blocked.Records[0].OutputDir = Record("blocked-cold", "follower").OutputDir;
            blocked.Records[1].SliceConfigIDs = "missing-plan";
            var active = Hospital(worker, "skip-active", 3, starts);
            active.Records[0].OutputDir = Record("blocked-cold", "other-follower").OutputDir;
            worker.DispatchHospitals(3, Hospital(worker, "skip-empty", 0, starts), blocked, active);
            Check(starts.SequenceEqual(new[] { "skip-active", "skip-active" })
                && Started(blocked.Records) == 0 && active.Records[0].ProcessID == -1
                && worker.LiveCount == 3, "Blocked hospital or blocked queue head stopped runnable work");
        }

        // Spawn failure uses no turn or capacity; the next runnable task from the
        // same hospital can start before moving on to the next hospital.
        using (Worker worker = new())
        {
            List<string> starts = new();
            var failed = Hospital(worker, "spawn-recovery-A", 2, starts);
            var other = Hospital(worker, "spawn-recovery-B", 2, starts);
            config.ExecutablePath = Path.Combine(Environment.CurrentDirectory, "missing-fairness-worker");
            failed.Context.Instance.ObserveUpdate += (record, columns) =>
            {
                if (record is AIFilmSliceRecord task && task.SliceImageStatus == (int)SliceImageStatus.切割失败)
                    config.ExecutablePath = executable;
            };
            worker.DispatchHospitals(2, failed, other);
            Check(failed.Records[0].SliceImageStatus == (int)SliceImageStatus.切割失败
                && starts.SequenceEqual(new[] { "spawn-recovery-A", "spawn-recovery-B" })
                && worker.LiveCount == 2, "Spawn failure blocked remaining hospitals or reserved capacity");
        }

        // Even persistently failing database writes after a real launch must use
        // that hospital's turn, preserve the child, and give the next turn to B.
        using (Worker worker = new())
        {
            List<string> starts = new();
            var a = Hospital(worker, "database-fair-A", 3, starts);
            var b = Hospital(worker, "database-fair-B", 3, starts);
            foreach (var record in a.Records) record.CacheDir = Path.Combine(Path.GetDirectoryName(record.OutputDir)!, "cache");
            a.Context.Instance.BeforeUpdate = () => throw new Exception("Persistent database outage after spawn");
            worker.DispatchHospitals(2, a, b);
            Check(starts.SequenceEqual(new[] { "database-fair-A", "database-fair-B" })
                && worker.LiveCount == 2, "Database failure lost the child or consumed another hospital's turn");
        }

        // A bad input combined with a database failure must also skip to B.
        using (Worker worker = new())
        {
            List<string> starts = new();
            var a = Hospital(worker, "bad-input-A", 1, starts);
            var b = Hospital(worker, "bad-input-B", 1, starts);
            a.Records[0].InputDir = Path.Combine(Environment.CurrentDirectory, "missing-input");
            a.Context.Instance.BeforeUpdate = () => throw new Exception("Cannot save failed input status");
            worker.DispatchHospitals(1, a, b);
            Check(starts.SequenceEqual(new[] { "bad-input-B" }) && worker.LiveCount == 1,
                "An invalid task and failed status write blocked another hospital");
        }

        TestCompleteBusinessScan();
        TestBusinessRecovery(executable);
        Console.WriteLine("Hospital fairness and complete-scan ordering passed");
    }

    static void TestStudyDateOrdering()
    {
        using Worker worker = new();
        List<string> startedExaminations = new();
        var hospital = Hospital(worker, "study-date", 3, new());
        AddStudy(hospital.Context, hospital.Records[0], "date-late", studyDate: new DateTime(2026, 1, 3));
        AddStudy(hospital.Context, hospital.Records[1], "date-early", studyDate: new DateTime(2026, 1, 1));
        AddStudy(hospital.Context, hospital.Records[2], "date-middle", studyDate: new DateTime(2026, 1, 2));
        hospital.Context.Instance.ObserveUpdate += (record, columns) =>
        {
            if (record is AIFilmSliceRecord task && task.ProcessID > 0)
                startedExaminations.Add(task.ExaminationID);
        };
        AISliceDicomServiceUtil.Instance.HospitalInfos = new() { hospital.Context.Hospital };
        DBServices.Databases[hospital.Context.Hospital.HospitalGroupID] = hospital.Context.Instance;
        AISliceDicomServiceUtil.Instance.AIFilmSliceConfigInfo.MaxConcurrentCount = 1;
        worker.RunBusiness();
        Check(startedExaminations.SequenceEqual(new[] { "date-early" }),
            "Non-urgent examinations were not dispatched in StudyDate order");
        AISliceDicomServiceUtil.Instance.HospitalInfos = new();
        DBServices.Databases.Clear();
    }

    static void TestCompleteBusinessScan()
    {
        using Worker worker = new();
        List<string> starts = new();
        var a = Hospital(worker, "business-A", 10, starts);
        var b = Hospital(worker, "business-B", 1, starts);
        var broken = Hospital(worker, "business-broken", 0, starts);
        // The urgent study is last in the database, but its preparation finishes
        // after a normal study. Dispatch must use urgency, not completion order.
        for (int i = 0; i < a.Records.Count; i++) AddStudy(a.Context, a.Records[i], $"normal-{i}", urgent: i == 9);
        AddStudy(b.Context, b.Records[0], "hospital-b");
        var contexts = new[] { a.Context, broken.Context, b.Context };
        AISliceDicomServiceUtil.Instance.HospitalInfos = contexts.Select(context => context.Hospital).ToList();
        foreach (var context in contexts) DBServices.Databases[context.Hospital.HospitalGroupID] = context.Instance;
        broken.Context.Instance.BeforeRead = type =>
        {
            if (type == typeof(V_ExaminationImage)) throw new Exception("One hospital's pending-view query failed");
        };
        int monitored = 0;
        using ManualResetEventSlim normalPrepared = new(false);
        worker.MonitorHospital = (records, context) =>
        {
            Check(worker.LiveCount == 0, "A hospital started children before all hospitals completed monitoring");
            if (records[0].ExaminationID == "normal-9")
                Check(normalPrepared.Wait(TimeSpan.FromSeconds(5)), "The normal study did not finish before the urgent study");
            else if (context.Hospital.ID == "business-A") normalPrepared.Set();
            Interlocked.Increment(ref monitored);
        };
        foreach (var context in new[] { a.Context, b.Context })
            context.Instance.ObserveUpdate += (record, columns) =>
            {
                if (record is AIFilmSliceRecord task && task.ProcessID > 0)
                    Check(monitored == 11, "Dispatch began before the complete cross-hospital scan finished");
            };
        AISliceDicomServiceUtil.Instance.AIFilmSliceConfigInfo.MaxConcurrentCount = 2;
        worker.RunBusiness();
        Check(monitored == 11 && starts.SequenceEqual(new[] { "business-A", "business-B" })
            && a.Records[9].ProcessID > 0 && a.Records.Take(9).All(record => record.ProcessID == -1),
            "Business scan failed to preserve urgent priority, hospital fairness, or per-hospital error isolation");
        // Monitoring still happens when every process slot remains occupied.
        worker.MonitorHospital = (records, context) => Interlocked.Increment(ref monitored);
        worker.RunBusiness();
        Check(monitored == 22 && starts.Count == 2 && worker.LiveCount == 2,
            "A saturated business scan skipped hospital monitoring or started extra children");
        AISliceDicomServiceUtil.Instance.HospitalInfos = new();
        DBServices.Databases.Clear();
    }

    static void TestBusinessRecovery(string executable)
    {
        using Worker worker = new();
        List<string> starts = new();
        var a = Hospital(worker, "restore-first", 1, starts);
        var b = Hospital(worker, "restore-last", 1, starts);
        using Process restored = Process.Start(new ProcessStartInfo
        {
            FileName = executable, UseShellExecute = false, CreateNoWindow = true,
        }) ?? throw new Exception("Could not start recovery child");
        b.Records[0].SliceImageStatus = (int)SliceImageStatus.正在切割;
        b.Records[0].ProcessID = restored.Id;
        b.Records[0].StartDate = restored.StartTime;
        AddStudy(a.Context, a.Records[0], "recovery-a");
        AddStudy(b.Context, b.Records[0], "recovery-b");
        AISliceDicomServiceUtil.Instance.HospitalInfos = new() { a.Context.Hospital, b.Context.Hospital };
        DBServices.Databases[a.Context.Hospital.ID] = a.Context.Instance;
        DBServices.Databases[b.Context.Hospital.ID] = b.Context.Instance;
        AISliceDicomServiceUtil.Instance.AIFilmSliceConfigInfo.MaxConcurrentCount = 1;
        int monitored = 0;
        worker.MonitorHospital = (records, context) =>
        {
            Check(worker.LiveCount == 1, "Startup recovery did not account for a later hospital's live process");
            monitored++;
        };
        worker.RunBusiness();
        Check(monitored == 2 && starts.Count == 0 && worker.LiveCount == 1 && a.Records[0].ProcessID == -1,
            "A late hospital's restored child did not reserve its global slot before dispatch");
        AISliceDicomServiceUtil.Instance.HospitalInfos = new();
        DBServices.Databases.Clear();
    }

    public static void Main(string[] args)
    {
        ThreadPool.SetMinThreads(8, 8);
        Config config = AISliceDicomServiceUtil.Instance.AIFilmSliceConfigInfo;
        config.ExecutablePath = args[0];

        // Exercise the actual service launcher and inspect its process arguments,
        // including a study path that must remain a single argument despite spaces.
        string studyParent = Path.Combine(Environment.CurrentDirectory, "check with spaces");
        Directory.CreateDirectory(studyParent);
        using (Worker worker = new())
        using (Process process = worker.LaunchForArguments(
            Path.Combine(studyParent, "series-A", "cache"),
            Path.Combine(studyParent, "series-A", "肺部")))
        {
            string[] arguments = process.StartInfo.ArgumentList.ToArray();
            int parentIndex = Array.IndexOf(arguments, "--series-cache-parent");
            Check(parentIndex >= 0 && arguments[parentIndex + 1] == studyParent,
                "Launcher did not pass the check parent as one argument");
            int cacheIndex = Array.IndexOf(arguments, "--cache_dir");
            Check(arguments[cacheIndex + 1] == Path.Combine(studyParent, "series-A", "cache"),
                "Parent propagation changed the per-series cache argument");
        }
        using (Worker worker = new())
        using (Process process = worker.LaunchForArguments(
            Path.Combine(Environment.CurrentDirectory, "missing-check", "series-A", "cache"),
            Path.Combine(Environment.CurrentDirectory, "missing-check", "series-A", "肺部")))
        {
            Check(!process.StartInfo.ArgumentList.Contains("--series-cache-parent"),
                "Launcher passed a nonexistent check parent instead of keeping the old invocation");
        }
        Console.WriteLine("Series reuse invocation arguments passed");

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
                Check(Started(followers) == 0 && worker.LiveCount == 0,
                    "Study monitoring started a child before completing the preparation phase");
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
        TestHospitalFairness(args[0]);
    }
}
