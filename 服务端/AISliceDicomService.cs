using ElincloudAISliceServer.DBEntity;
using ElincloudAISliceServer.Entity;
using ElincloudServer.Common;
using ElincloudServer.CommonBLService;
using ElincloudServer.DBService;
using ElincloudServer.Entity;
using Newtonsoft.Json;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.ServiceModel.Channels;
using ExaminationImage = ElincloudAISliceServer.DBEntity.ExaminationImage;
using RISExamination = ElincloudAISliceServer.DBEntity.RISExamination;

namespace ElincloudAISliceServer.AISliceDicomService
{
    public class AISliceDicomWorker : BackgroundService
    {
        private readonly ILogger<AISliceDicomWorker> _logger;

        private static readonly DateTime MinStartDate = new DateTime(1900, 1, 1);

        /// <summary>
        /// 执行结果文件名称
        /// </summary>
        private const string ResultFileName = "report.json";

        public AISliceDicomWorker(ILogger<AISliceDicomWorker> logger)
        {
            _logger = logger;
        }

        public override Task StartAsync(CancellationToken cancellationToken)
        {
            if (AISliceDicomServiceUtil.Instance.InitSuccess)
            {
                return base.StartAsync(cancellationToken);
            }
            else
            {
                return Task.CompletedTask;
            }
        }

        /// <summary>
        /// 是否能执行
        /// </summary>
        /// <returns></returns>
        private async Task<bool> needCalc()
        {
            await Task.Delay(AISliceDicomServiceUtil.Instance.ExecuteWaitMinute);
            return true;
        }

        /// <summary>
        /// Python进程跟踪锁；并发数来自真实存活进程，不按任务记录累加或递减。
        /// </summary>
        private readonly object ProcessNumLock = new object();

        // PID保存实际启动时间：相同进程的多条记录只计一次，也能识别PID复用。
        private readonly Dictionary<int, DateTime> RunningProcesses = new Dictionary<int, DateTime>();
        // 不同部位可并行，但同一输出目录只能由现有任务持有；保留恢复时已重复的全部PID。
        private readonly Dictionary<string, Dictionary<int, DateTime>> RunningOutputs =
            new Dictionary<string, Dictionary<int, DateTime>>(OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal);
        private bool RunningProcessesRestored;

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            AISliceDicomServiceUtil.Instance.PrintStatusLogAction($"启动成功！");

            while (!stoppingToken.IsCancellationRequested)
            {
                AISliceDicomServiceUtil.Instance.PrintLogAction($"正在运行！");
                try
                {
                    if (await needCalc())
                    {
                        BusinessLogic();
                    }
                }
                catch (Exception ex)
                {
                    AISliceDicomServiceUtil.Instance.PrintLogAction($"执行异常!", true, null, ex);
                }
                finally
                {
                    await Task.Delay(10000, stoppingToken);
                }
            }
        }

        private void BusinessLogic()
        {
            // 恢复全部医院的旧进程后才能派发新任务，避免前一个医院先占满新名额。
            RestoreRunningProcesses();
            AISliceDicomServiceUtil.Instance.PrintLogAction($"共有[{AISliceDicomServiceUtil.Instance.HospitalInfos.Count}]个医院的数据需要处理！");
            foreach (HospitalInfo hospital in AISliceDicomServiceUtil.Instance.HospitalInfos)
            {
                AISliceDicomServiceUtil.Instance.PrintLogAction($"开始处理[{hospital.Name}]的数据！");
                #region 处理单个医院
                SqlAccess instance = DBServices.GetHospitalDBInstance(hospital.HospitalGroupID);

                List<SlicePlanConfigInfo> slicePlans = CommonBL.GetSlicePlanConfigInfos(hospital.HospitalGroupID);
                List<string> studyTypes = slicePlans.Select(i => i.StudyTypeName).ToList();

                List<RISExamination> list = instance.GetMuch<RISExamination>(i =>
                    i.Status == (int)ExaminationStatus.已检查
                    && studyTypes.Contains(i.StudyTypeName)
                    && (i.SliceImageStatus == (int)SliceImageStatus.未切割 || i.SliceImageStatus == (int)SliceImageStatus.正在切割)
                );

                if (list != null && list.Count > 0)
                {
                    AISliceDicomServiceUtil.Instance.PrintLogAction($"[{hospital.Name}]需要处理{list.Count()}条数据！");
                    // 每次循环处理的数量
                    int handleCount = 1;
                    // 循环次数
                    int loopCount = (int)Math.Ceiling((double)list.Count / handleCount);
                    // 大循环
                    for (int i = 0; i < loopCount; i++)
                    {
                        // 获取本次循环要处理的数据
                        List<RISExamination> sublist = list.Skip(handleCount * i).Take(handleCount).ToList();
                        // 开始本次小循环
                        List<Task> subTasks = new List<Task>();
                        foreach (var item in sublist)
                        {
                            System.Threading.Tasks.Task subtask = new System.Threading.Tasks.Task(new Action(() =>
                            {
                                ProcessSingleStudy(item, slicePlans, instance, hospital);
                                AISliceDicomServiceUtil.Instance.PrintLogAction($"检查号[{item.ID}]的数据处理完成！");
                            }));
                            subTasks.Add(subtask);
                            subtask.Start();
                        }
                        //等待线程池结束,执行下一次循环
                        Task.WaitAll(subTasks.ToArray());
                    }
                    AISliceDicomServiceUtil.Instance.PrintLogAction($"[{hospital.Name}]的数据处理完成！");
                }
                else
                {
                    AISliceDicomServiceUtil.Instance.PrintLogAction($"[{hospital.Name}]没有要处理的数据！");
                }
                #endregion
            }
        }

        /// <summary>
        /// 处理单个检查数据
        /// </summary>
        /// <param name="examination"></param>
        private void ProcessSingleStudy(RISExamination examination, List<SlicePlanConfigInfo> slicePlans, SqlAccess instance, HospitalInfo hospital)
        {
            try
            {
                SliceImageStatus status = ElincloudServer.Common.Utils.EnumUtil.GetEnumValue<SliceImageStatus>(examination.SliceImageStatus);
                switch (status)
                {
                    case SliceImageStatus.未切割:
                        ProcessUnprocessedStudy(examination, slicePlans, instance, hospital);
                        break;
                    case SliceImageStatus.正在切割:
                        ProcessProcessingStudy(examination, slicePlans, instance);
                        break;
                }
            }
            catch (Exception ex)
            {
                AISliceDicomServiceUtil.Instance.PrintLogAction($"检查号[{examination.ID}]的数据处理异常！", true, null, ex);
            }
        }

        /// <summary>
        /// 处理一条未切割检查。
        /// </summary>
        /// <param name="examination">当前待处理的 RIS 检查。</param>
        /// <param name="slicePlans">切割方案。</param>
        /// <param name="instance">当前医院的数据库访问对象。</param>
        private void ProcessUnprocessedStudy(RISExamination examination, List<SlicePlanConfigInfo> slicePlans, SqlAccess instance, HospitalInfo hospital)
        {
            // 当前检查的根路径
            string examinationoutputDir = ElincloudServer.Common.Utils.StringUtil.Combine(AISliceDicomServiceUtil.Instance.ServiceCommonConfigInfo.FileStoragePath, AISliceDicomServiceUtil.Instance.ServiceCommonConfigInfo.FileCacheFolder, ApplicationEx.AISliceImageCacheFolder, hospital.ID, examination.ID);

            // 查出所有序列
            List<ExaminationImage> seriesList = instance.GetMuch<ExaminationImage>(item => item.ExaminationID == examination.ImageExaminationID && item.PatientID == examination.ImagePatientID && item.IsArchived == (int)DataStoreStatus.未归档 && item.Category == (int)DicomImageCategories.原始影像);

            // 检查状态更新失败或服务中断时，序列记录可能已经写入。
            // 这里先查询已有记录，防止同一序列重复启动Python和重复写入记录。
            List<AIFilmSliceRecord> existingRecords = instance.GetMuch<AIFilmSliceRecord>(item => item.ExaminationID == examination.ID && item.PatientID == examination.PatientID);

            bool slice = existingRecords.Count > 0;
            foreach (ExaminationImage series in seriesList)
            {
                float scanThickness = 0f;
                float.TryParse(series?.ScanThickness, out scanThickness);

                string seriesInstanceUID = series.SeriesInstanceUID;
                string studyType = series.StudyTypeName;

                // 拼接序列路径，即输入路径
                string inputDir = ElincloudServer.Common.Utils.StringUtil.Combine(AISliceDicomServiceUtil.Instance.ServiceCommonConfigInfo.FileStoragePath, series.ImageStoreDirectory, series.ImageRelaPath);

                if (series.SeriesImageCount == 0)
                {
                    #region 更新数量
                    if (Directory.Exists(inputDir))
                    {
                        series.SeriesImageCount = Directory.GetFiles(inputDir).Length;
                        instance.Update<ExaminationImage>(series, i => i.SeriesInstanceUID == series.SeriesInstanceUID, ExaminationImage._SeriesImageCount);
                    }
                    #endregion
                }

                string sourceStudyPartName = series.BodyPartExamined?.Trim() ?? string.Empty;

                AIFilmSliceRecord record = new AIFilmSliceRecord()
                {
                    ID = instance.GetID<AIFilmSliceRecord>(),
                    PatientID = examination.PatientID,
                    ExaminationID = examination.ID,
                    SeriesInstanceUID = seriesInstanceUID,
                    StudyTypeName = studyType,
                    ScanThickness = scanThickness,
                };

                BodyPartMapping? mapping = AISliceDicomServiceUtil.Instance.AIFilmSliceConfigInfo.BodyPartMappings?
                                                                    .FirstOrDefault(i => string.Equals(i.SeriesStudyPartTag?.Trim(), sourceStudyPartName, StringComparison.OrdinalIgnoreCase));

                // 无映射部位
                if (mapping == null || string.IsNullOrWhiteSpace(mapping.StudyPartName))
                {
                    record.SliceImageStatus = (int)SliceImageStatus.不切割;
                    record.Message = $"检查部位[{sourceStudyPartName}]    层厚[{scanThickness}mm]   未映射到检查部位，跳过！";
                    instance.Insert<AIFilmSliceRecord>(record);

                    AISliceDicomServiceUtil.Instance.PrintLogAction($"检查号[{examination.ID}]  序列[{seriesInstanceUID}]${record.Message}");
                    continue;
                }

                string studyPartName = mapping.StudyPartName.Trim();

                // 匹配是否有合适的切割方案
                List<SlicePlanConfigInfo> matchedPlans = slicePlans.FindAll(i =>
                                                                                                            i.StudyTypeName == studyType
                                                                                                            && i.StudyPartName == studyPartName
                                                                                                            && scanThickness >= i.MinScanThickness
                                                                                                            && scanThickness <= i.MaxScanThickness
                                                                                                        );

                // 无切割方案
                if (matchedPlans.Count == 0)
                {
                    record.SliceImageStatus = (int)SliceImageStatus.不切割;
                    record.Message = $"检查部位[{sourceStudyPartName}]   层厚[{scanThickness}mm]   没有对应切割方案，跳过！";
                    instance.Insert<AIFilmSliceRecord>(record);

                    AISliceDicomServiceUtil.Instance.PrintLogAction($"检查号[{examination.ID}]  序列[{seriesInstanceUID}]${record.Message}");
                    continue;
                }

                // 一个序列可能命中多个切割部位；每个部位单独调用一次 Python，
                // 但共用 cacheDir，从而复用 NIfTI 和模型分割缓存。
                foreach (var slicePartGroup in matchedPlans.GroupBy(item => item.SlicePartName, StringComparer.OrdinalIgnoreCase))
                {
                    List<SlicePlanConfigInfo> currentPlans = slicePartGroup.ToList();
                    if (existingRecords.Any(item =>
                        item.SeriesInstanceUID == seriesInstanceUID
                        && string.Equals(item.SlicePartName, slicePartGroup.Key, StringComparison.OrdinalIgnoreCase)))
                    {
                        continue;
                    }
                    string cacheDir = ElincloudServer.Common.Utils.StringUtil.Combine(examinationoutputDir, seriesInstanceUID, "cache");
                    // 输出路径
                    string outputDir = ElincloudServer.Common.Utils.StringUtil.Combine(examinationoutputDir, seriesInstanceUID, slicePartGroup.Key);

                    // 转换成切割服务要的结构
                    List<PythonPlaneConfig> planeConfigs = currentPlans
                                                                                        .GroupBy(config => new { config.Plane, config.SliceType, config.SliceThickness, config.SliceSpacing, config.SliceImageCount, config.FillPercent })
                                                                                        .Select(group => new PythonPlaneConfig
                                                                                        {
                                                                                            Plane = group.Key.Plane.ToString(),
                                                                                            SliceType = group.Key.SliceType.ToString(),
                                                                                            SliceThickness = group.Key.SliceThickness,
                                                                                            SliceSpacing = group.Key.SliceSpacing,
                                                                                            SliceImageCount = group.Key.SliceImageCount,
                                                                                            FillPercent = group.Key.FillPercent,
                                                                                            WindowPresets = group.Select(config => config.WindowPreset.Trim()).Distinct(StringComparer.OrdinalIgnoreCase).ToList(),
                                                                                        })
                                                                                        .ToList();
                    string planeConfigJson = JsonConvert.SerializeObject(planeConfigs);

                    int processID = -1;
                    DateTime startDate = MinStartDate;
                    SliceImageStatus sliceImageStatus = SliceImageStatus.未切割;
                    string message = string.Empty;

                    // DICOM目录不存在时不启动Python，但仍然保存该序列的失败记录，
                    // 便于后续通过记录表查看具体失败原因。
                    if (!Directory.Exists(inputDir))
                    {
                        processID = 0;
                        sliceImageStatus = SliceImageStatus.切割失败;
                        message = $"DICOM目录不存在：{inputDir}";
                        AISliceDicomServiceUtil.Instance.PrintLogAction($"检查号[{examination.ID}]序列[{seriesInstanceUID}]{message}", true);
                    }
                    // 共享缓存允许同序列并行，所有切割部位共同遵守真实进程总数上限。
                    else if (CanStartProcess(AISliceDicomServiceUtil.Instance.AIFilmSliceConfigInfo.MaxConcurrentCount, outputDir))
                    {
                        try
                        {
                            using Process process = StartAutoPrintFilm(inputDir, cacheDir, outputDir, studyType, slicePartGroup.Key, scanThickness, planeConfigJson, BuildWindowPresetJson(currentPlans));
                            processID = process.Id;
                            startDate = process.StartTime;   // 用真实进程启动时间
                            sliceImageStatus = SliceImageStatus.正在切割;
                        }
                        catch (Exception ex)
                        {
                            // 启动后即使数据库写入失败，仍按真实PID保留名额直到退出。
                            RefreshRunningProcesses();
                            processID = 0;
                            sliceImageStatus = SliceImageStatus.切割失败;
                            message = ex.Message;
                            AISliceDicomServiceUtil.Instance.PrintLogAction($"检查号[{examination.ID}]序列[{seriesInstanceUID}]启动Python失败！", true, null, ex);
                        }
                    }
                    else
                    {
                        // 当前Python进程数量已达上限时，并未执行失败，
                        // 只保存为“未切割”，等待后续在“正在切割”逻辑中重新启动。
                        processID = -1;
                        sliceImageStatus = SliceImageStatus.未切割;
                        message = string.Empty;
                    }

                    record.ID = instance.GetID<AIFilmSliceRecord>();
                    record.SliceImageStatus = (int)sliceImageStatus;
                    record.Message = message;
                    record.SlicePath = Path.GetRelativePath(AISliceDicomServiceUtil.Instance.ServiceCommonConfigInfo.FileStoragePath, outputDir);
                    record.StudyPartName = studyPartName;
                    record.SlicePartName = slicePartGroup.Key;
                    record.ProcessID = processID;
                    record.SliceConfigIDs = string.Join(",", currentPlans.Select(i => i.ID));
                    record.InputDir = inputDir;
                    record.CacheDir = cacheDir;
                    record.OutputDir = outputDir;
                    record.PlaneConfigJson = planeConfigJson;
                    record.CreateDate = DateTime.Now;
                    if (sliceImageStatus == SliceImageStatus.正在切割)
                    {
                        record.StartDate = startDate;
                    }

                    slice = true;

                    instance.Insert<AIFilmSliceRecord>(record);
                }
            }

            // “正在切割”，等待后续轮询所有序列的 report.json。
            examination.SliceImageStatus = slice ? (int)SliceImageStatus.正在切割 : (int)SliceImageStatus.不切割;
            instance.Update<RISExamination>(examination, item => item.ID == examination.ID, RISExamination._SliceImageStatus);
        }

        /// <summary>
        /// 处理一条正在切割的检查
        /// </summary>
        /// <param name="examination">当前正在处理的RIS检查</param>
        /// <param name="slicePlans">切割方案</param>
        /// <param name="instance">当前医院的数据库访问对象</param>
        private void ProcessProcessingStudy(RISExamination examination, List<SlicePlanConfigInfo> slicePlans, SqlAccess instance)
        {
            AIDicomSliceConfigInfo serviceConfig = AISliceDicomServiceUtil.Instance.AIFilmSliceConfigInfo;

            // 获取当前检查的切割记录
            List<AIFilmSliceRecord> records = instance.GetMuch<AIFilmSliceRecord>(item => item.ExaminationID == examination.ID && item.PatientID == examination.PatientID);
            if (records.Count == 0)
            {
                // 检查处于“正在切割”但没有任何切割记录时，还原为“未切割”等待重新分析。
                examination.SliceImageStatus = (int)SliceImageStatus.未切割;
                instance.Update<RISExamination>(examination, item => item.ID == examination.ID, RISExamination._SliceImageStatus);
                return;
            }

            foreach (AIFilmSliceRecord record in records)
            {
                // 已切割 切割失败的直接跳过
                if (record.SliceImageStatus == (int)SliceImageStatus.已切割 || record.SliceImageStatus == (int)SliceImageStatus.切割失败)
                {
                    continue;
                }

                List<SlicePlanConfigInfo> recordSlecePlan = slicePlans.FindAll(i => record.SliceConfigIDs.Split(",").Contains(i.ID));

                string reportFile = ElincloudServer.Common.Utils.StringUtil.Combine(record.OutputDir, ResultFileName);

                // 报告可能先于进程退出生成，必须等真实进程退出后再结束当前任务。
                if (IsProcessRunning(record.ProcessID, record.StartDate) || IsOutputProcessRunning(record.OutputDir))
                {
                    continue;
                }

                // 进程已退出，优先处理已经生成的报告。
                if (File.Exists(reportFile))
                {
                    ProcessSliceReport(reportFile, record, recordSlecePlan, instance);
                    RefreshRunningProcesses();
                    continue;
                }

                // 进程仍然存在则继续等待；进程已经消失则整组还原为“未切割”。
                if (record.SliceImageStatus == (int)SliceImageStatus.正在切割)
                {
                    if (IsProcessRunning(record.ProcessID, record.StartDate))
                    {
                        continue;
                    }

                    UpdateSeriesRecordStatus(record, SliceImageStatus.未切割, -1, MinStartDate, string.Empty, instance);
                    RefreshRunningProcesses();

                    // 进程消失后本轮只还原状态，不立即重新启动，等待下一轮处理。
                    continue;
                }

                // 序列状态为“未切割”时，说明之前并发数量已满或旧进程已经消失，
                // 每条记录按可用的全局名额启动，不按序列限制并发。
                if (record.SliceImageStatus == (int)SliceImageStatus.未切割)
                {
                    StartUnprocessedSeries(record, recordSlecePlan, serviceConfig.MaxConcurrentCount, instance);
                }
            }

            // 检查表只记录当前检查是否已经全部处理完成。
            // 当前检查任何序列无论是“已切割”还是“切割失败”都属于终态。
            if (records.All(item => item.SliceImageStatus == (int)SliceImageStatus.不切割 || item.SliceImageStatus == (int)SliceImageStatus.已切割 || item.SliceImageStatus == (int)SliceImageStatus.切割失败))
            {
                examination.SliceImageStatus = (int)SliceImageStatus.已切割;
                instance.Update<RISExamination>(examination, item => item.ID == examination.ID, RISExamination._SliceImageStatus);
            }
        }

        /// <summary>
        /// 启动处于“未切割”状态的序列记录
        /// </summary>
        /// <param name="record">切割记录</param>
        /// <param name="maxConcurrentCount">允许同时运行的最大Python进程数量</param>
        /// <param name="instance">当前医院的数据库访问对象</param>
        private void StartUnprocessedSeries(AIFilmSliceRecord record, List<SlicePlanConfigInfo> slicePlans, int maxConcurrentCount, SqlAccess instance)
        {
            if (!Directory.Exists(record.InputDir))
            {
                string message = $"DICOM目录不存在：{record.InputDir}";
                UpdateSeriesRecordStatus(record, SliceImageStatus.切割失败, 0, MinStartDate, message, instance);
                AISliceDicomServiceUtil.Instance.PrintLogAction($"检查号[{record.ExaminationID}]序列[{record.SeriesInstanceUID}]{message}", true);
                return;
            }

            // 全局名额已满或同一输出仍有任务运行时等待；不同部位仍可并行。
            if (!CanStartProcess(maxConcurrentCount, record.OutputDir))
            {
                return;
            }

            try
            {
                using Process process = StartAutoPrintFilm(record.InputDir, record.CacheDir, record.OutputDir, record.StudyTypeName, record.SlicePartName, record.ScanThickness, record.PlaneConfigJson, BuildWindowPresetJson(slicePlans));
                UpdateSeriesRecordStatus(record, SliceImageStatus.正在切割, process.Id, process.StartTime, string.Empty, instance);
            }
            catch (Exception ex)
            {
                RefreshRunningProcesses();
                UpdateSeriesRecordStatus(record, SliceImageStatus.切割失败, 0, MinStartDate, ex.Message, instance);
                AISliceDicomServiceUtil.Instance.PrintLogAction($"检查号[{record.ExaminationID}]序列[{record.SeriesInstanceUID}]启动Python失败！", true, null, ex);
            }
        }

        /// <summary>
        /// 处理一个序列的report.json，并把合并后的Python结果按切割方案拆分入库
        /// </summary>
        /// <param name="reportFile">report.json文件路径</param>
        /// <param name="record">切割记录</param>
        /// <param name="slicePlans">切割方案列表</param>
        /// <param name="instance">当前医院的数据库访问对象</param>
        private void ProcessSliceReport(string reportFile, AIFilmSliceRecord record, List<SlicePlanConfigInfo> slicePlans, SqlAccess instance)
        {
            DateTime updateDate = DateTime.Now;
            PythonSliceReportInfo report;
            try
            {
                string reportJson = File.ReadAllText(reportFile);
                report = JsonConvert.DeserializeObject<PythonSliceReportInfo>(reportJson) ?? throw new InvalidOperationException("report.json内容为空");
            }
            catch (Exception ex)
            {
                string message = $"report.json解析失败：{ex.Message}";
                UpdateReportFailedRecords(record, message, instance);
                AISliceDicomServiceUtil.Instance.PrintLogAction($"检查号[{record.ExaminationID}]序列[{record.SeriesInstanceUID}]{message}", true, null, ex);
                return;
            }

            if (!report.Success)
            {
                string message = string.IsNullOrWhiteSpace(report.Message) ? "Python切割执行失败" : report.Message;
                UpdateReportFailedRecords(record, message, instance);
                return;
            }

            Transaction tran = instance.GetTransaction();
            try
            {
                List<AIFilmSliceResult> results = GetSliceResults(record, slicePlans, report, instance);
                if (results.Count == 0)
                {
                    throw new InvalidOperationException("报告中没有找到对应的切割结果");
                }

                tran.Insert<AIFilmSliceResult>(results.ToArray());
                record.SliceImageStatus = (int)SliceImageStatus.已切割;
                record.Message = string.Empty;
            }
            catch (Exception ex)
            {
                record.SliceImageStatus = (int)SliceImageStatus.切割失败;
                record.Message = ex.Message;
                AISliceDicomServiceUtil.Instance.PrintLogAction($"检查号[{record.ExaminationID}]序列[{record.SeriesInstanceUID}]拆分切割结果失败！", true, null, ex);
            }
            record.UpdateDate = updateDate;
            tran.Update<AIFilmSliceRecord>(record, item => item.ID == record.ID, AIFilmSliceRecord._SliceImageStatus, AIFilmSliceRecord._Message, AIFilmSliceRecord._UpdateDate);

            if (!tran.Commit())
            {
                throw new InvalidOperationException($"序列[{record.SeriesInstanceUID}]保存切割结果失败！");
            }
        }

        /// <summary>
        /// 从Python合并报告中拆分切割结果
        /// </summary>
        /// <param name="record">切割记录</param>
        /// <param name="slicePlans">切割方案</param>
        /// <param name="report">Python返回的切割报告</param>
        /// <param name="instance">当前医院的数据库访问对象</param>
        /// <returns>当前模板需要保存的所有切割结果</returns>
        private static List<AIFilmSliceResult> GetSliceResults(AIFilmSliceRecord record, List<SlicePlanConfigInfo> slicePlans, PythonSliceReportInfo report, SqlAccess instance)
        {
            List<AIFilmSliceResult> results = new List<AIFilmSliceResult>();
            foreach (SlicePlanConfigInfo sliceConfig in slicePlans)
            {
                PythonSliceReportPlaneInfo planeInfo = report.PlaneList.FirstOrDefault(item =>
                    string.Equals(item.Plane, sliceConfig.Plane.ToString(), StringComparison.OrdinalIgnoreCase)
                    && string.Equals(item.SliceType, sliceConfig.SliceType.ToString(), StringComparison.OrdinalIgnoreCase)
                    && (sliceConfig.SliceThickness == 0 ? !item.SliceThickness.HasValue || item.SliceThickness.Value == 0 : item.SliceThickness.HasValue && Math.Abs(item.SliceThickness.Value - sliceConfig.SliceThickness) < 0.0001f)
                    && (sliceConfig.SliceSpacing == 0 ? !item.SliceSpacing.HasValue || item.SliceSpacing.Value == 0 : item.SliceSpacing.HasValue && Math.Abs(item.SliceSpacing.Value - sliceConfig.SliceSpacing) < 0.0001f)
                    && (sliceConfig.SliceImageCount == 0 ? !item.SliceImageCount.HasValue || item.SliceImageCount.Value == 0 : item.SliceImageCount.HasValue && item.SliceImageCount.Value == sliceConfig.SliceImageCount)
                    && (sliceConfig.FillPercent == 0 ? !item.FillPercent.HasValue || item.FillPercent.Value == 0 : item.FillPercent.HasValue && item.FillPercent.Value == sliceConfig.FillPercent)
                    );
                if (planeInfo == null)
                {
                    throw new InvalidOperationException($"报告中未找到[{sliceConfig.Plane}][{sliceConfig.SliceType}]对应的切割配置");
                }

                PythonSliceReportWindowInfo windowInfo = planeInfo.WindowPresetList.FirstOrDefault(item => string.Equals(item.Name, sliceConfig.WindowPreset, StringComparison.OrdinalIgnoreCase));
                if (windowInfo == null || windowInfo.RegionList.Count == 0)
                {
                    throw new InvalidOperationException($"报告中未找到窗位[{sliceConfig.WindowPreset}]对应的切割结果");
                }

                foreach (PythonSliceReportRegionInfo regionInfo in windowInfo.RegionList)
                {
                    if (!Enum.TryParse(regionInfo.Region, true, out SliceRegions sliceRegion))
                    {
                        throw new InvalidOperationException($"报告中存在未支持的切割区域[{regionInfo.Region}]");
                    }

                    AIFilmSliceResult result = new AIFilmSliceResult()
                    {
                        ID = instance.GetID<AIFilmSliceResult>(),
                        AIFilmSliceRecordID = record.ID,
                        SlicePlanConfigID = sliceConfig.ID,
                        SliceRegion = (int)sliceRegion,
                        SliceThickness = regionInfo.ActualSliceThickness,
                        SliceSpacing = regionInfo.ActualSliceSpacing,
                        SliceImageCount = regionInfo.ActualSliceImageCount,
                        ImagePath = Path.GetRelativePath(AISliceDicomServiceUtil.Instance.ServiceCommonConfigInfo.FileStoragePath, regionInfo.Path),
                    };
                    results.Add(result);
                }
            }
            return results;
        }

        /// <summary>
        /// 更新切割记录状态
        /// </summary>
        /// <param name="record">切割记录</param>
        /// <param name="sliceImageStatus">目标切割状态</param>
        /// <param name="processID">Python进程ID</param>
        /// <param name="startDate">Python启动时间</param>
        /// <param name="message">错误原因</param>
        /// <param name="instance">当前医院的数据库访问对象</param>
        private static void UpdateSeriesRecordStatus(AIFilmSliceRecord record, SliceImageStatus sliceImageStatus, int processID, DateTime startDate, string message, SqlAccess instance)
        {
            try
            {
                record.SliceImageStatus = (int)sliceImageStatus;
                record.ProcessID = processID;
                record.StartDate = startDate;
                record.Message = message;
                instance.Update<AIFilmSliceRecord>(record, item => item.ID == record.ID, AIFilmSliceRecord._SliceImageStatus, AIFilmSliceRecord._ProcessID, AIFilmSliceRecord._StartDate, AIFilmSliceRecord._Message);
            }
            catch (Exception)
            {
                throw new InvalidOperationException($"序列[{record.SeriesInstanceUID}]更新切割记录状态失败！");
            }
        }

        /// <summary>
        /// report.json执行失败或格式错误时，将序列对应的记录改为处理失败
        /// </summary>
        /// <param name="record">切割记录</param>
        /// <param name="message">失败原因</param>
        /// <param name="instance">当前医院的数据库访问对象</param>
        private static void UpdateReportFailedRecords(AIFilmSliceRecord record, string message, SqlAccess instance)
        {
            try
            {
                DateTime updateDate = DateTime.Now;
                record.SliceImageStatus = (int)SliceImageStatus.切割失败;
                record.Message = message;
                record.UpdateDate = updateDate;
                instance.Update<AIFilmSliceRecord>(record, item => item.ID == record.ID, AIFilmSliceRecord._SliceImageStatus, AIFilmSliceRecord._Message, AIFilmSliceRecord._UpdateDate);
            }
            catch (Exception)
            {
                throw new InvalidOperationException($"序列[{record.SeriesInstanceUID}]保存失败状态失败！");
            }
        }


        #region 非业务方法
        /// <summary>
        /// 判断指定的操作系统进程是否仍然存在
        /// </summary>
        /// <param name="processID">操作系统进程ID</param>
        /// <returns>进程存在返回true，进程不存在返回false</returns>
        private static bool IsProcessRunning(int processID, DateTime? expectedStartTime = null)
        {
            return TryGetRunningProcessStartTime(processID, expectedStartTime, out _);
        }

        private static bool TryGetRunningProcessStartTime(int processID, DateTime? expectedStartTime, out DateTime startTime)
        {
            startTime = MinStartDate;
            if (processID <= 0) return false;
            try
            {
                using Process process = Process.GetProcessById(processID);
                if (process.HasExited) return false;
                startTime = process.StartTime;
                return !expectedStartTime.HasValue || expectedStartTime.Value == MinStartDate
                    || Math.Abs((startTime - expectedStartTime.Value).TotalSeconds) <= 5;
            }
            catch (ArgumentException) { return false; }
            catch (InvalidOperationException) { return false; }
        }

        /// <summary>
        /// 首次派发前恢复全部医院的存活PID；部分数据库不可读时不派发新任务。
        /// </summary>
        private void RestoreRunningProcesses()
        {
            if (RunningProcessesRestored) return;
            foreach (HospitalInfo hospital in AISliceDicomServiceUtil.Instance.HospitalInfos)
            {
                SqlAccess instance = DBServices.GetHospitalDBInstance(hospital.HospitalGroupID);
                List<AIFilmSliceRecord> records = instance.GetMuch<AIFilmSliceRecord>(item =>
                    item.SliceImageStatus == (int)SliceImageStatus.正在切割 && item.ProcessID > 0);
                foreach (AIFilmSliceRecord record in records)
                {
                    if (TryGetRunningProcessStartTime(record.ProcessID, record.StartDate, out DateTime startTime))
                    {
                        TrackRunningProcess(record.ProcessID, startTime, record.OutputDir);
                    }
                }
            }
            RefreshRunningProcesses();
            RunningProcessesRestored = true;
        }

        /// <summary>
        /// 当前业务循环按检查串行派发（handleCount = 1），启动成功后立即登记真实PID。
        /// </summary>
        private bool CanStartProcess(int maxConcurrentCount, string outputDir)
        {
            RestoreRunningProcesses();
            string outputKey = NormalizeOutputDirectory(outputDir);
            lock (ProcessNumLock)
            {
                RefreshRunningProcesses();
                return RunningProcesses.Count < maxConcurrentCount && !RunningOutputs.ContainsKey(outputKey);
            }
        }

        private static string NormalizeOutputDirectory(string outputDir)
        {
            return Path.TrimEndingDirectorySeparator(Path.GetFullPath(outputDir));
        }

        private bool IsOutputProcessRunning(string outputDir)
        {
            if (string.IsNullOrWhiteSpace(outputDir)) return false;
            RestoreRunningProcesses();
            string outputKey = NormalizeOutputDirectory(outputDir);
            lock (ProcessNumLock)
            {
                RefreshRunningProcesses();
                return RunningOutputs.ContainsKey(outputKey);
            }
        }

        private void TrackRunningProcess(int processID, DateTime startTime, string outputDir)
        {
            string outputKey = string.IsNullOrWhiteSpace(outputDir) ? null : NormalizeOutputDirectory(outputDir);
            lock (ProcessNumLock)
            {
                RunningProcesses[processID] = startTime;
                if (outputKey != null)
                {
                    if (!RunningOutputs.TryGetValue(outputKey, out Dictionary<int, DateTime> processes))
                    {
                        RunningOutputs[outputKey] = processes = new Dictionary<int, DateTime>();
                    }
                    processes[processID] = startTime;
                }
            }
        }

        private void RefreshRunningProcesses()
        {
            lock (ProcessNumLock)
            {
                foreach (var item in RunningProcesses.ToList())
                {
                    if (!IsProcessRunning(item.Key, item.Value))
                    {
                        RunningProcesses.Remove(item.Key);
                    }
                }
                foreach (var output in RunningOutputs.ToList())
                {
                    foreach (var process in output.Value.ToList())
                    {
                        if (!RunningProcesses.TryGetValue(process.Key, out DateTime startTime) || startTime != process.Value)
                        {
                            output.Value.Remove(process.Key);
                        }
                    }
                    if (output.Value.Count == 0) RunningOutputs.Remove(output.Key);
                }
            }
        }

        /// <summary>
        /// 根据切割方案生成本次 Python 请求使用的窗宽窗位列表。
        /// </summary>
        private static string BuildWindowPresetJson(IEnumerable<SlicePlanConfigInfo> slicePlans)
        {
            HashSet<string> names = slicePlans
                .Select(plan => plan.WindowPreset.Trim())
                .Where(name => !string.IsNullOrWhiteSpace(name))
                .ToHashSet(StringComparer.OrdinalIgnoreCase);

            List<object> presets = AISliceDicomServiceUtil.Instance.AIFilmSliceConfigInfo.WindowPresets
                .Where(preset => names.Contains(preset.Name))
                .Select(preset => new
                {
                    Name = preset.Name,
                    windowWidth = preset.WindowWidth,
                    windowLevel = preset.WindowLevel,
                })
                .Cast<object>()
                .ToList();
            return JsonConvert.SerializeObject(presets);
        }

        /// <summary>
        /// 启动一个序列的 AutoPrintFilm Python 进程，但不等待进程结束。
        /// </summary>
        /// <returns>启动成功后的操作系统进程 ID。</returns>
        private Process StartAutoPrintFilm(string inputDir, string cacheDir, string outDir, string studyType, string slicePart, double scanThickness, string planeConfigJson, string windowPresetJson)
        {
            string executablePath = Path.GetFullPath(
                AISliceDicomServiceUtil.Instance.AIFilmSliceConfigInfo.ExecutablePath);
            string outputKey = NormalizeOutputDirectory(outDir);

            ProcessStartInfo startInfo = new ProcessStartInfo
            {
                FileName = executablePath,
                WorkingDirectory = Path.GetDirectoryName(executablePath)!,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            startInfo.ArgumentList.Add("--input");
            startInfo.ArgumentList.Add(inputDir);
            startInfo.ArgumentList.Add("--cache_dir");
            startInfo.ArgumentList.Add(cacheDir);
            startInfo.ArgumentList.Add("--output");
            startInfo.ArgumentList.Add(outDir);
            startInfo.ArgumentList.Add("--slice_part");
            startInfo.ArgumentList.Add(slicePart);
            startInfo.ArgumentList.Add("--study_type");
            startInfo.ArgumentList.Add(studyType);
            startInfo.ArgumentList.Add("--scan_thickness");
            startInfo.ArgumentList.Add(scanThickness.ToString(CultureInfo.InvariantCulture));
            startInfo.ArgumentList.Add("--plane_configs");
            startInfo.ArgumentList.Add(planeConfigJson);
            startInfo.ArgumentList.Add("--window_presets");
            startInfo.ArgumentList.Add(windowPresetJson);

            Process process = Process.Start(startInfo)
                ?? throw new InvalidOperationException("AutoPrintFilm进程启动失败");

            // 先跟踪真实进程，再返回调用方写数据库；写入失败也不会漏计活进程。
            TrackRunningProcess(process.Id, MinStartDate, outputKey);
            try
            {
                DateTime startDate = process.StartTime;
                TrackRunningProcess(process.Id, startDate, outputKey);
                return process;
            }
            catch
            {
                process.Dispose();
                throw;
            }
        }
        #endregion
    }
}
