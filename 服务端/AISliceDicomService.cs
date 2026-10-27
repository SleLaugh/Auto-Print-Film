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
        // 同序列首任务结束且已有正式缓存后，不同部位可并行；同一输出目录始终独占。
        // 保留恢复时已重复的全部PID，避免漏计旧服务已经启动的进程。
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
        /// 处理单个检查数据。
        /// 按照“补齐缺失任务 -> 启动待处理任务 -> 监测任务完成情况”的顺序执行，
        /// 本轮出现新的成功或失败终态时，再次派发以放行已有缓存的同序列剩余任务。
        /// 未切割和正在切割的检查都需要检查是否存在尚未创建任务的原始序列。
        /// </summary>
        /// <param name="examination">当前待处理的 RIS 检查。</param>
        /// <param name="slicePlans">切割方案。</param>
        /// <param name="instance">当前医院的数据库访问对象。</param>
        /// <param name="hospital">当前医院信息。</param>
        private void ProcessSingleStudy(RISExamination examination, List<SlicePlanConfigInfo> slicePlans, SqlAccess instance, HospitalInfo hospital)
        {
            try
            {
                SliceImageStatus originalStatus = ElincloudServer.Common.Utils.EnumUtil.GetEnumValue<SliceImageStatus>(examination.SliceImageStatus);
                // 第一步：扫描原始序列，为尚未创建任务的序列补建任务记录。
                // 返回的记录集合在后两个阶段复用，避免为了拆分方法重复查询任务表。
                List<AIFilmSliceRecord> records = CreateMissingSeriesTasks(examination, slicePlans, instance, hospital);
                // 第二步：启动已经创建、当前仍处于“未切割”状态的任务。
                StartUnprocessedTasks(records, slicePlans, instance);
                // 第三步：基于任务记录监测进程、处理报告并更新检查整体状态。
                int completedCount = records.Count(record => record.SliceImageStatus == (int)SliceImageStatus.已切割
                    || record.SliceImageStatus == (int)SliceImageStatus.切割失败);
                MonitorTaskCompletion(examination, records, slicePlans, instance, originalStatus);
                if (records.Count(record => record.SliceImageStatus == (int)SliceImageStatus.已切割
                    || record.SliceImageStatus == (int)SliceImageStatus.切割失败) > completedCount)
                {
                    // 切割失败也可能已完成分割缓存，同轮重新派发，不再多等一个扫描周期。
                    StartUnprocessedTasks(records, slicePlans, instance);
                }
            }
            catch (Exception ex)
            {
                AISliceDicomServiceUtil.Instance.PrintLogAction($"检查号[{examination.ID}]的数据处理异常！", true, null, ex);
            }
        }

        /// <summary>
        /// 创建缺失的序列切割任务。
        /// 只负责扫描原始 DICOM 序列、比对已有任务记录并创建缺失记录，不在此方法中启动 Python。
        /// 未切割和正在切割的检查都会调用此方法，以便发现检查处理过程中后来出现的序列。
        /// </summary>
        /// <param name="examination">当前待处理的 RIS 检查。</param>
        /// <param name="slicePlans">切割方案。</param>
        /// <param name="instance">当前医院的数据库访问对象。</param>
        /// <param name="hospital">当前医院信息。</param>
        /// <returns>本轮检查的全部切割任务记录，包括原有记录和本次新建的记录。</returns>
        private List<AIFilmSliceRecord> CreateMissingSeriesTasks(RISExamination examination, List<SlicePlanConfigInfo> slicePlans, SqlAccess instance, HospitalInfo hospital)
        {
            // 当前检查的根路径
            string examinationoutputDir = ElincloudServer.Common.Utils.StringUtil.Combine(AISliceDicomServiceUtil.Instance.ServiceCommonConfigInfo.FileStoragePath, AISliceDicomServiceUtil.Instance.ServiceCommonConfigInfo.FileCacheFolder, ApplicationEx.AISliceImageCacheFolder, hospital.ID, examination.ID);

            // 查出所有序列
            List<ExaminationImage> seriesList = instance.GetMuch<ExaminationImage>(item => item.ExaminationID == examination.ImageExaminationID && item.PatientID == examination.ImagePatientID && item.IsArchived == (int)DataStoreStatus.未归档 && item.Category == (int)DicomImageCategories.原始影像);

            // 检查状态更新失败或服务中断时，序列记录可能已经写入。
            // 先查询已有记录，后续直接复用这个集合，防止重复创建任务。
            List<AIFilmSliceRecord> records = instance.GetMuch<AIFilmSliceRecord>(item => item.ExaminationID == examination.ID && item.PatientID == examination.PatientID);
            if (records == null)
            {
                records = new List<AIFilmSliceRecord>();
            }
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

                // 数量小于10，不做任何处理
                if (series.SeriesImageCount < 10)
                {
                    record.SliceImageStatus = (int)SliceImageStatus.不切割;
                    record.Message = $"检查部位[{sourceStudyPartName}]    层厚[{scanThickness}mm]   影像数量不到10张，跳过！";
                    instance.Insert<AIFilmSliceRecord>(record);
                    records.Add(record);
                    AISliceDicomServiceUtil.Instance.PrintLogAction($"检查号[{examination.ID}]  序列[{seriesInstanceUID}]${record.Message}");
                    continue;
                }

                BodyPartMapping? mapping = AISliceDicomServiceUtil.Instance.AIFilmSliceConfigInfo.BodyPartMappings?
                                                                    .FirstOrDefault(i => string.Equals(i.SeriesStudyPartTag?.Trim(), sourceStudyPartName, StringComparison.OrdinalIgnoreCase));

                // 无映射部位：同一序列已经有“不切割”记录时，不再重复插入。
                if (mapping == null || string.IsNullOrWhiteSpace(mapping.StudyPartName))
                {
                    if (records.Any(item => item.SeriesInstanceUID == seriesInstanceUID && item.SliceImageStatus == (int)SliceImageStatus.不切割))
                    {
                        continue;
                    }
                    record.SliceImageStatus = (int)SliceImageStatus.不切割;
                    record.Message = $"检查部位[{sourceStudyPartName}]    层厚[{scanThickness}mm]   未映射到检查部位，跳过！";
                    instance.Insert<AIFilmSliceRecord>(record);
                    records.Add(record);
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

                // 无切割方案：同一序列已经有“不切割”记录时，不再重复插入。
                if (matchedPlans.Count == 0)
                {
                    if (records.Any(item => item.SeriesInstanceUID == seriesInstanceUID && item.SliceImageStatus == (int)SliceImageStatus.不切割))
                    {
                        continue;
                    }
                    record.SliceImageStatus = (int)SliceImageStatus.不切割;
                    record.Message = $"检查部位[{sourceStudyPartName}]   层厚[{scanThickness}mm]   没有对应切割方案，跳过！";
                    instance.Insert<AIFilmSliceRecord>(record);
                    records.Add(record);
                    AISliceDicomServiceUtil.Instance.PrintLogAction($"检查号[{examination.ID}]  序列[{seriesInstanceUID}]${record.Message}");
                    continue;
                }

                // 一个序列可能命中多个切割部位；每个部位单独调用一次 Python，
                // 同序列共用缓存，各部位任务可在全局名额内并行。
                foreach (var slicePartGroup in matchedPlans.GroupBy(item => item.SlicePartName, StringComparer.OrdinalIgnoreCase))
                {
                    List<SlicePlanConfigInfo> currentPlans = slicePartGroup.ToList();
                    if (records.Any(item =>
                        item.SeriesInstanceUID == seriesInstanceUID
                        && string.Equals(item.SlicePartName, slicePartGroup.Key, StringComparison.OrdinalIgnoreCase)))
                    {
                        continue;
                    }
                    // 输出按部位隔离；同序列共用缓存，缓存独立于会被清理的输出目录。
                    string outputDir = ElincloudServer.Common.Utils.StringUtil.Combine(examinationoutputDir, seriesInstanceUID, slicePartGroup.Key);
                    string cacheDir = GetSeriesCacheDirectory(outputDir);

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
                    SliceImageStatus sliceImageStatus = SliceImageStatus.未切割;
                    string message = string.Empty;

                    // 创建阶段只创建任务记录，不启动 Python。
                    // DICOM目录不存在时直接保存失败记录，便于后续通过记录表查看具体失败原因。
                    if (!Directory.Exists(inputDir))
                    {
                        processID = 0;
                        sliceImageStatus = SliceImageStatus.切割失败;
                        message = $"DICOM目录不存在：{inputDir}";
                        AISliceDicomServiceUtil.Instance.PrintLogAction($"检查号[{examination.ID}]序列[{seriesInstanceUID}]{message}", true);
                    }

                    // 每个切割部位都必须使用独立的记录对象，避免一个序列命中多个部位时复用同一个对象。
                    AIFilmSliceRecord taskRecord = new AIFilmSliceRecord()
                    {
                        ID = instance.GetID<AIFilmSliceRecord>(),
                        PatientID = examination.PatientID,
                        ExaminationID = examination.ID,
                        SeriesInstanceUID = seriesInstanceUID,
                        StudyTypeName = studyType,
                        ScanThickness = scanThickness,
                        SliceImageStatus = (int)sliceImageStatus,
                        Message = message,
                        SlicePath = Path.GetRelativePath(AISliceDicomServiceUtil.Instance.ServiceCommonConfigInfo.FileStoragePath, outputDir),
                        StudyPartName = studyPartName,
                        SlicePartName = slicePartGroup.Key,
                        ProcessID = processID,
                        SliceConfigIDs = string.Join(",", currentPlans.Select(i => i.ID)),
                        InputDir = inputDir,
                        CacheDir = cacheDir,
                        OutputDir = outputDir,
                        PlaneConfigJson = planeConfigJson,
                        CreateDate = DateTime.Now,
                    };
                    instance.Insert<AIFilmSliceRecord>(taskRecord);
                    records.Add(taskRecord);
                }
            }

            return records;
        }

        /// <summary>
        /// 启动本轮已经创建、处于“未切割”状态的任务。
        /// 此方法不扫描原始 DICOM 序列，只根据任务记录启动任务。
        /// </summary>
        /// <param name="records">当前检查的全部切割任务记录。</param>
        /// <param name="slicePlans">切割方案列表。</param>
        /// <param name="instance">当前医院的数据库访问对象。</param>
        private void StartUnprocessedTasks(List<AIFilmSliceRecord> records, List<SlicePlanConfigInfo> slicePlans, SqlAccess instance)
        {
            int maxConcurrentCount = AISliceDicomServiceUtil.Instance.AIFilmSliceConfigInfo.MaxConcurrentCount;

            foreach (AIFilmSliceRecord record in records)
            {
                // 只启动“未切割”状态的任务；已运行、已完成、失败和不切割的记录均不在此处启动。
                if (record.SliceImageStatus != (int)SliceImageStatus.未切割)
                {
                    continue;
                }

                // 任务配置 ID 可能为空（例如不切割记录），先判断状态后再解析配置。
                List<SlicePlanConfigInfo> recordSlicePlans = string.IsNullOrWhiteSpace(record.SliceConfigIDs)
                    ? new List<SlicePlanConfigInfo>()
                    : slicePlans.FindAll(i => record.SliceConfigIDs.Split(',').Contains(i.ID));

                if (recordSlicePlans.Count == 0)
                {
                    // 正常情况下，未切割任务必须具有切割方案；避免空配置启动 Python。
                    continue;
                }

                StartUnprocessedSeries(record, recordSlicePlans, maxConcurrentCount, instance, records);
            }
        }

        /// <summary>
        /// 基于任务记录监测任务完成情况。
        /// 只监测已有任务，不扫描原始序列，也不负责启动待处理任务。
        /// </summary>
        /// <param name="examination">当前正在处理的 RIS 检查。</param>
        /// <param name="records">当前检查的全部切割任务记录。</param>
        /// <param name="slicePlans">切割方案列表。</param>
        /// <param name="instance">当前医院的数据库访问对象。</param>
        /// <param name="originalStatus">本轮处理前的检查状态。</param>
        private void MonitorTaskCompletion(RISExamination examination, List<AIFilmSliceRecord> records, List<SlicePlanConfigInfo> slicePlans, SqlAccess instance, SliceImageStatus originalStatus)
        {
            if (records.Count == 0)
            {
                // 正在切割但没有任何任务记录时，还原为“未切割”，等待下一轮重新分析；
                // 原本未切割且没有可创建任务的序列时，则标记为“不切割”。
                examination.SliceImageStatus = originalStatus == SliceImageStatus.正在切割
                    ? (int)SliceImageStatus.未切割
                    : (int)SliceImageStatus.不切割;
                instance.Update<RISExamination>(
                    examination, item => item.ID == examination.ID, RISExamination._SliceImageStatus);
                return;
            }

            foreach (AIFilmSliceRecord record in records)
            {
                // 已切割、切割失败和不切割都是终态，不需要继续监测。
                if (record.SliceImageStatus == (int)SliceImageStatus.已切割
                    || record.SliceImageStatus == (int)SliceImageStatus.切割失败
                    || record.SliceImageStatus == (int)SliceImageStatus.不切割)
                {
                    continue;
                }

                List<SlicePlanConfigInfo> recordSlicePlans = string.IsNullOrWhiteSpace(record.SliceConfigIDs)
                    ? new List<SlicePlanConfigInfo>()
                    : slicePlans.FindAll(i => record.SliceConfigIDs.Split(',').Contains(i.ID));

                string reportFile = ElincloudServer.Common.Utils.StringUtil.Combine(record.OutputDir, ResultFileName);

                // 报告可能先于进程退出生成，必须等真实进程退出后再结束当前任务。
                if (IsProcessRunning(record.ProcessID, record.StartDate) || IsOutputProcessRunning(record.OutputDir))
                {
                    continue;
                }

                // 进程已退出，优先处理已经生成的报告。
                if (File.Exists(reportFile))
                {
                    ProcessSliceReport(reportFile, record, recordSlicePlans, instance);
                    RefreshRunningProcesses();
                    continue;
                }

                // 进程已经退出且没有报告时，将任务还原为“未切割”，下一轮再尝试启动。
                // 本轮不在监测阶段重新启动任务，保持“创建、启动、监测”三个阶段职责分离。
                if (record.SliceImageStatus == (int)SliceImageStatus.正在切割)
                {
                    UpdateSeriesRecordStatus(record, SliceImageStatus.未切割, -1, MinStartDate, string.Empty, instance);
                    RefreshRunningProcesses();
                }
            }

            // 检查表只记录当前检查是否已经全部处理完成。
            // “不切割”“已切割”和“切割失败”均属于任务终态。
            bool allFinished = records.All(item =>
                item.SliceImageStatus == (int)SliceImageStatus.不切割
                || item.SliceImageStatus == (int)SliceImageStatus.已切割
                || item.SliceImageStatus == (int)SliceImageStatus.切割失败);

            if (allFinished)
            {
                bool allNotCut = records.All(item => item.SliceImageStatus == (int)SliceImageStatus.不切割);
                bool anyCut = records.Any(item => item.SliceImageStatus == (int)SliceImageStatus.已切割);
                bool anyFailed = records.Any(item => item.SliceImageStatus == (int)SliceImageStatus.切割失败);

                // 规则：
                // 1. 全部不切割 -> 不切割
                // 2. 只要存在已切割 -> 已切割（即使同时有失败/不切割）
                // 3. 没有已切割，但存在切割失败（含“不切割+切割失败”混合、全部失败）-> 切割失败
                SliceImageStatus finalStatus;
                if (allNotCut)
                {
                    finalStatus = SliceImageStatus.不切割;
                }
                else if (anyCut)
                {
                    finalStatus = SliceImageStatus.已切割;
                }
                else if (anyFailed)
                {
                    finalStatus = SliceImageStatus.切割失败;
                }
                else
                {
                    // 理论上不会走到这里：allFinished 且非 allNotCut、无已切割、无失败，
                    // 只可能是 records 为空，但前面已处理 records.Count == 0。
                    finalStatus = SliceImageStatus.不切割;
                }

                examination.SliceImageStatus = (int)finalStatus;
                instance.Update<RISExamination>(
                    examination, item => item.ID == examination.ID, RISExamination._SliceImageStatus);
            }
            else if (examination.SliceImageStatus != (int)SliceImageStatus.正在切割)
            {
                // 只要还有任务未进入终态（未切割/正在切割），检查整体状态就应保持“正在切割”。
                // 例如并发名额已满时，任务记录仍是“未切割”，但检查已经进入任务处理流程。
                // 特别注意：只要存在“正在切割”，即使同时已有“已切割”，检查也保持“正在切割”。
                examination.SliceImageStatus = (int)SliceImageStatus.正在切割;
                instance.Update<RISExamination>(examination, item => item.ID == examination.ID, RISExamination._SliceImageStatus);
            }
        }
        /// <summary>
        /// 启动处于“未切割”状态的序列记录；首任务结束且已有正式缓存后放行同序列并发。
        /// </summary>
        /// <param name="record">切割记录</param>
        /// <param name="maxConcurrentCount">允许同时运行的最大Python进程数量</param>
        /// <param name="instance">当前医院的数据库访问对象</param>
        /// <param name="seriesRecords">当前检查的全部任务，用于识别同序列已结束的任务</param>
        private void StartUnprocessedSeries(AIFilmSliceRecord record, List<SlicePlanConfigInfo> slicePlans, int maxConcurrentCount, SqlAccess instance, IEnumerable<AIFilmSliceRecord>? seriesRecords = null)
        {
            if (!Directory.Exists(record.InputDir))
            {
                string message = $"DICOM目录不存在：{record.InputDir}";
                UpdateSeriesRecordStatus(record, SliceImageStatus.切割失败, 0, MinStartDate, message, instance);
                AISliceDicomServiceUtil.Instance.PrintLogAction($"检查号[{record.ExaminationID}]序列[{record.SeriesInstanceUID}]{message}", true);
                return;
            }

            try
            {
                string cacheDir = GetSeriesCacheDirectory(record.OutputDir);
                // 同序列任务继续复用同一缓存；先保存旧记录的缓存迁移，失败时不启动新进程。
                // 数据库写入在调度锁外，之后由启动方法原子检查名额、输出目录及序列占用。
                if (!string.Equals(record.CacheDir, cacheDir, OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal))
                {
                    record.CacheDir = cacheDir;
                    instance.Update<AIFilmSliceRecord>(record, item => item.ID == record.ID, nameof(AIFilmSliceRecord.CacheDir));
                }

                // 成片成功与缓存生成是两个条件：缺体素等后续失败仍可能留下可复用缓存。
                // 先确认同序列有已结束任务，启动锁内再检查缓存；不根据错误消息判断。
                StringComparison pathComparison = OperatingSystem.IsWindows()
                    ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
                bool allowSeriesConcurrency = seriesRecords != null && seriesRecords.Any(completed =>
                    (completed.SliceImageStatus == (int)SliceImageStatus.已切割
                        || completed.SliceImageStatus == (int)SliceImageStatus.切割失败)
                    && !string.IsNullOrWhiteSpace(completed.OutputDir)
                    && string.Equals(GetSeriesCacheDirectory(completed.OutputDir), cacheDir, pathComparison)
                    && !IsOutputProcessRunning(completed.OutputDir));

                using Process? process = StartAutoPrintFilm(maxConcurrentCount, record.InputDir, cacheDir, record.OutputDir, record.StudyTypeName, record.SlicePartName, record.ScanThickness, record.PlaneConfigJson, BuildWindowPresetJson(slicePlans), allowSeriesConcurrency);
                // 名额已满、输出目录被占用或同序列首任务仍在运行时保留待处理，继续扫描其他任务。
                if (process == null)
                {
                    return;
                }

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
        /// 输出为“序列/部位”，缓存固定为“序列/cache”，重建任务记录后仍可复用。
        /// </summary>
        private static string GetSeriesCacheDirectory(string outputDir)
        {
            string outputPath = NormalizeOutputDirectory(outputDir);
            string seriesDirectory = Path.GetDirectoryName(outputPath)
                ?? throw new InvalidOperationException("无法根据输出目录确定序列缓存路径");
            return Path.Combine(seriesDirectory, "cache");
        }

        /// <summary>
        /// 判断序列是否已有正式发布的原始体数据和至少一个模型的分割缓存。
        /// 不读取图像内容；沿用 Python 的文件与模型标记判定，忽略私有工作目录。
        /// </summary>
        private static bool HasReusableSeriesCache(string cacheDir)
        {
            try
            {
                if (!File.Exists(Path.Combine(cacheDir, "nifti", "raw_0000.nii.gz"))) return false;

                foreach (string modelDir in Directory.EnumerateDirectories(cacheDir))
                {
                    string modelID = Path.GetFileName(modelDir);
                    if (!int.TryParse(modelID, NumberStyles.None, CultureInfo.InvariantCulture, out int datasetID)
                        || datasetID <= 0) continue;

                    string segmentationDir = Path.Combine(modelDir, "segmentation");
                    string segmentationPath = Path.Combine(segmentationDir, "seg.nii.gz");
                    if (!File.Exists(segmentationPath)) continue;

                    string markerPath = Path.Combine(segmentationDir, "model.json");
                    // 兼容没有模型标记的历史缓存，正式分割文件是最终发布标志。
                    if (!File.Exists(markerPath)) return File.Exists(segmentationPath);

                    try
                    {
                        using System.Text.Json.JsonDocument marker = System.Text.Json.JsonDocument.Parse(File.ReadAllText(markerPath));
                        if (marker.RootElement.ValueKind != System.Text.Json.JsonValueKind.Object
                            || !marker.RootElement.TryGetProperty("model_id", out System.Text.Json.JsonElement model)) continue;
                        string? markedID = model.ValueKind == System.Text.Json.JsonValueKind.String ? model.GetString()
                            : model.ValueKind == System.Text.Json.JsonValueKind.Number ? model.GetRawText() : null;
                        if (string.Equals(markedID, modelID, StringComparison.Ordinal) && File.Exists(segmentationPath))
                        {
                            return true;
                        }
                    }
                    catch (IOException) { }
                    catch (UnauthorizedAccessException) { }
                    catch (System.Text.Json.JsonException) { }
                }
            }
            // 暂时读不到缓存时继续保守派发，不将任务误记为切割失败。
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
            return false;
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
        /// 原子检查全局名额、输出目录及同序列占用，启动并登记真实PID，不等待进程结束。
        /// 同序列已有结束任务且正式缓存已生成后，不同输出目录可按全局上限并行。
        /// </summary>
        /// <returns>启动成功后的进程；名额、输出目录或同序列首任务限制不允许启动时返回null。</returns>
        private Process? StartAutoPrintFilm(int maxConcurrentCount, string inputDir, string cacheDir, string outDir, string studyType, string slicePart, double scanThickness, string planeConfigJson, string windowPresetJson, bool allowSeriesConcurrency = false)
        {
            // 首次恢复涉及数据库读取；业务循环在并发派发前已完成恢复。
            RestoreRunningProcesses();
            string executablePath = Path.GetFullPath(
                AISliceDicomServiceUtil.Instance.AIFilmSliceConfigInfo.ExecutablePath);
            string outputKey = NormalizeOutputDirectory(outDir);
            string cacheKey = NormalizeOutputDirectory(cacheDir);

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

            lock (ProcessNumLock)
            {
                RefreshRunningProcesses();
                // 检查、启动和登记必须持同一把锁，避免并发派发抢占同一名额或首任务资格。
                if (RunningProcesses.Count >= maxConcurrentCount || RunningOutputs.ContainsKey(outputKey))
                {
                    return null;
                }
                // 等待任务留在服务端队列，不启动 Python 占用进程和全局并发名额。
                // 首任务成功或失败都可放行，但必须已有正式缓存；缺失缓存时仅允许单个接替。
                StringComparison pathComparison = OperatingSystem.IsWindows()
                    ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
                if ((!allowSeriesConcurrency || !HasReusableSeriesCache(cacheKey)) && RunningOutputs.Keys.Any(runningOutput =>
                    string.Equals(GetSeriesCacheDirectory(runningOutput), cacheKey, pathComparison)))
                {
                    return null;
                }

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
        }
        #endregion
    }
}
