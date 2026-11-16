using ElincloudServer.Common;
using ElincloudServer.Entity;
using DCMImageInfo = ElincloudAISliceServer.Entity.DCMImageInfo;

namespace ElincloudServer.ArchiveService
{
    public class ArchiveDBWorker : BackgroundService
    {
        private readonly ILogger<ArchiveDBWorker> _logger;
        private string childServerName = "数据库写入服务";

        public ArchiveDBWorker(ILogger<ArchiveDBWorker> logger)
        {
            _logger = logger;
        }

        public override Task StartAsync(CancellationToken cancellationToken)
        {
            if (ArchiveConfigUtil.Instance.InitSuccess)
            {
                return base.StartAsync(cancellationToken);
            }
            else
            {
                return Task.CompletedTask;
            }
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            if (System.IO.Directory.Exists(ArchiveConfigUtil.Instance.ImageInfoPath))
            {
                ArchiveConfigUtil.Instance.PrintStatusLogAction($"启动成功！归档数据目录[{ArchiveConfigUtil.Instance.ImageInfoPath}]!", childServerName);
                int handleCount = ArchiveConfigUtil.Instance.IsEnableArchiveCache ? ArchiveConfigUtil.Instance.ArchiveCacheFileCount : 1;
                while (!stoppingToken.IsCancellationRequested)
                {
                    try
                    {
                        ArchiveConfigUtil.Instance.PrintLogAction($"正在运行！", false, childServerName);
                        MoveFailedFiles();

                        List<string> files = System.IO.Directory.GetFiles(ArchiveConfigUtil.Instance.ImageInfoPath).ToList().FindAll(p => (DateTime.Now - System.IO.File.GetCreationTime(p)).TotalSeconds > 2);
                        if (files.Count > 0)
                        {
                            ArchiveConfigUtil.Instance.PrintLogAction($"开始处理归档数据...文件总数: [{files.Count}]个", false, childServerName);

                            int loopCount = (int)Math.Ceiling((double)files.Count / handleCount);
                            int remainingTotal = files.Count;
                            for (int i = 0; i < loopCount; i++)
                            {

                                System.Collections.Concurrent.ConcurrentDictionary<string, string> fileJsonDic = new System.Collections.Concurrent.ConcurrentDictionary<string, string>();
                                List<string> subFiles = files.Skip(handleCount * i).Take(handleCount).ToList();
                                ManualResetEvent[] waitEvents = new ManualResetEvent[subFiles.Count];
                                foreach (var file in subFiles)
                                {
                                    int index = subFiles.IndexOf(file);
                                    waitEvents[index] = new ManualResetEvent(false);
                                    ArchiveConfigUtil.Instance.PushActionToThreadPool(new WaitCallback((object obj) =>
                                    {
                                        try
                                        {
                                            string json = System.IO.File.ReadAllText(file);
                                            fileJsonDic.TryAdd(file, json);
                                        }
                                        catch (Exception ex)
                                        {
                                            ArchiveConfigUtil.Instance.PrintLogAction($"读取dii文件[{file}]异常!", true, childServerName, ex);
                                            ArchiveFailed(file);
                                        }
                                        finally
                                        {
                                            ManualResetEvent e = (ManualResetEvent)obj;
                                            e.Set();
                                        }
                                    }), waitEvents[index]);
                                }
                                //等待线程池结束,执行下一次循环
                                WaitHandle.WaitAll(waitEvents);

                                foreach (var item in fileJsonDic)
                                {
                                    try
                                    {
                                        bool success = AddDCMImageInfoToDB(item.Value);
                                        if (success)
                                        {
                                            //一个循环组结束统一插入detail
                                            success = AddArchiveDetialsToDB();
                                            if (success)
                                            {
                                                if (!ArchiveConfigUtil.Instance.IsEnableArchiveCache)
                                                {
                                                    //文件处理成功则清理该数据文件
                                                    System.IO.File.Delete(item.Key);
                                                }
                                            }
                                            else
                                            {
                                                ArchiveFailed(item.Key);
                                            }
                                        }
                                        else
                                        {
                                            ArchiveFailed(item.Key);
                                        }
                                    }
                                    catch (Exception ex)
                                    {
                                        ArchiveConfigUtil.Instance.PrintLogAction($"处理归档数据文件[{item.Key}]异常!", true, childServerName, ex);
                                        ArchiveFailed(item.Key);
                                    }
                                }
                                if (ArchiveConfigUtil.Instance.IsEnableArchiveCache)
                                {
                                    bool success = ArchiveConfigUtil.Instance.ArchiveBLInstance.AddCacheDataToDB(ArchiveConfigUtil.Instance.GetArchiveConfig, ArchiveConfigUtil.Instance.GetStudyType);
                                    if (success)
                                    {
                                        foreach (var item in fileJsonDic)
                                        {
                                            File.Delete(item.Key);
                                        }
                                    }
                                    else
                                    {
                                        foreach (var item in fileJsonDic)
                                        {
                                            ArchiveFailed(item.Key);
                                        }
                                    }
                                }
                                ArchiveConfigUtil.Instance.PrintLogAction(string.Format("正在处理归档数据...运行状态: {0}/{1}", remainingTotal, files.Count), false, childServerName);
                            }
                            ArchiveConfigUtil.Instance.PrintLogAction($"监听数据文件夹...空闲", false, childServerName);
                        }
                        else
                        {
                            ArchiveConfigUtil.Instance.PrintLogAction($"监听数据文件夹...空闲", false, childServerName);
                            await Task.Delay(1000, stoppingToken);
                        }
                    }
                    catch (Exception ex)
                    {
                        ArchiveConfigUtil.Instance.PrintLogAction($"执行异常!", true, childServerName, ex);
                    }
                }
            }
            else
            {
                if (ArchiveConfigUtil.Instance.ImageInfoPath == string.Empty)
                {
                    ArchiveConfigUtil.Instance.PrintLogAction($"启动失败！归档数据文件夹路径未配置！", true, childServerName);
                }
                else
                {
                    ArchiveConfigUtil.Instance.PrintLogAction($"启动失败！归档数据文件夹路径[{ArchiveConfigUtil.Instance.ImageInfoPath}]不存在", true, childServerName);
                }
            }
        }

        /// <summary>
        /// 将ErrorFiles中创建时间在2天内的文件移动到归档池中再次进行处理
        /// </summary>
        private void MoveFailedFiles()
        {
            try
            {
                string errorPath = Utils.StringUtil.Combine(ArchiveConfigUtil.Instance.ImageInfoPath, "ErrorFiles");
                if (System.IO.Directory.Exists(errorPath))
                {
                    List<string> files = System.IO.Directory.GetFiles(errorPath).ToList().FindAll(p => (DateTime.Now - System.IO.File.GetCreationTime(p)).TotalMinutes > 1 && (DateTime.Now - System.IO.File.GetCreationTime(p)).TotalMinutes < 60);
                    foreach (var file in files)
                    {
                        string targetFileName = Common.Utils.StringUtil.Combine(ArchiveConfigUtil.Instance.ImageInfoPath, Utils.StringUtil.GetFileName(file));
                        System.IO.File.Move(file, targetFileName);
                    }
                }
            }
            catch (Exception ex)
            {
                ArchiveConfigUtil.Instance.PrintLogAction($"移动ErrorFiles中的dii文件异常!", true, childServerName, ex);
            }
        }

        #region 数据库操作

        private double maxMinites = 5;
        internal static MultiDictionary<string, DCMImageInfo> archiveDetails = new MultiDictionary<string, DCMImageInfo>();
        private Dictionary<string, ArchiveCache> dics = new Dictionary<string, ArchiveCache>();

        private bool AddDCMImageInfoToDB(string json)
        {
            //if (System.IO.File.Exists(file))
            //{
            //    string json = System.IO.File.ReadAllText(file);
            MultiDictionary<string, DCMImageInfo> dcmInfoDics = ElincloudServer.Common.Utils.JsonUtil.FromJson<MultiDictionary<string, DCMImageInfo>>(json);
            if (dcmInfoDics != null)
            {
                foreach (var key in dcmInfoDics.Keys)
                {
                    List<DCMImageInfo> images = dcmInfoDics[key] as List<DCMImageInfo>;
                    if (images.Count > 0)
                    {
                        ArchiveConfigInfo archiveConfig = ArchiveConfigUtil.Instance.GetArchiveConfig(key);
                        foreach (var image in images)
                        {
                            bool success = FinishArchive(image, archiveConfig);
                            //但凡有一条数据更新数据库失败，这个文件则被认为处理失败
                            if (!success)
                            {
                                return success;
                            }
                        }
                    }
                }
            }
            //}
            return true;
        }

        private bool FinishArchive(DCMImageInfo info, ArchiveConfigInfo archiveConfig)
        {
            lock (dics)
            {
                //清除20分钟未使用的检查号
                try
                {
                    List<string> keys = new List<string>();
                    foreach (string key in dics.Keys)
                    {
                        ArchiveCache ac = dics[key];
                        if ((DateTime.Now - ac.CacheDate).TotalMinutes > maxMinites)
                        {
                            keys.Add(key);
                        }
                    }
                    foreach (var key in keys)
                    {
                        dics.Remove(key);
                    }
                }
                catch (Exception ex)
                {
                    ArchiveConfigUtil.Instance.PrintLogAction($"清除归档缓存检查号异常!", true, childServerName, ex);
                }

                bool success = false;
                string keyStr = info.ImageRelaPath;
                try
                {
                    //如果已经缓存过检查号说明这个检查的状态已经更新
                    if (!dics.ContainsKey(keyStr))
                    {
                        //归档
                        CheckInfo studyType = ArchiveConfigUtil.Instance.GetStudyType(info.HospitalNumber, info.StudyTypeName);
                        ArchiveConfigUtil.Instance.ArchiveBLInstance.FinishArchive(info, archiveConfig, studyType);
                        ArchiveCache ac = new ArchiveCache()
                        {
                            AccessNumber = info.ExaminationID,
                            PatientID = info.PatientID,
                            CacheDate = DateTime.Now
                        };
                        dics.Add(keyStr, ac);

                        if (!archiveConfig.IsEnableArchiveCache)
                        {
                            //自动生成患者云影像
                            if (archiveConfig.AutoGenerateStudy)
                            {
                                ArchiveConfigUtil.Instance.ArchiveBLInstance.AutoGenerateStudy(info, archiveConfig, studyType);
                            }
                        }
                        //自动生成云影像
                        if (archiveConfig.GenerateCloudImageReport)
                        {
                            ArchiveConfigUtil.Instance.ArchiveBLInstance.AutoGenerateCloudImageReport(info, archiveConfig, studyType);
                        }
                        success = true;
                    }
                    else
                    {
                        success = true;
                    }
                }
                catch (Exception ex)
                {
                    ArchiveConfigUtil.Instance.PrintLogAction($"归档后更新检查状态异常!影像号:[{info.PatientID}],检查号:[{info.ExaminationID}]", true, childServerName, ex);
                    if (dics.ContainsKey(keyStr))
                    {
                        dics.Remove(keyStr);
                    }
                    success = false;
                }


                //插入归档详情信息
                try
                {
                    if (archiveConfig.IsAddArchiveDetail)
                    {
                        archiveDetails.Add(archiveConfig.Hospital.HospitalGroupID, info);
                    }
                }
                catch (Exception ex)
                {
                    ArchiveConfigUtil.Instance.PrintLogAction($"插入归档详情信息失败!", true, childServerName, ex);
                }

                return success;
            }
        }

        internal bool AddArchiveDetialsToDB()
        {
            bool success = false;
            try
            {
                foreach (var key in archiveDetails.Keys)
                {
                    List<DCMImageInfo> images = archiveDetails[key] as List<DCMImageInfo>;
                    ArchiveConfigInfo archiveConfig = ArchiveConfigUtil.Instance.GetArchiveConfig(key);
                    ArchiveConfigUtil.Instance.ArchiveBLInstance.AddArchiveDetails(key, images, ArchiveConfigUtil.Instance.IsEnableArchiveCache);
                }
                success = true;
            }
            catch (Exception ex)
            {
                ArchiveConfigUtil.Instance.PrintLogAction($"批量插入detail数据异常!", true, childServerName, ex);
                success = false;
            }
            finally
            {
                archiveDetails.Clear();
            }
            return success;
        }
        private void ArchiveFailed(string fullFileName)
        {
            try
            {
                string errorFileName = Utils.StringUtil.Combine(ArchiveConfigUtil.Instance.ImageInfoErrorPath, Utils.StringUtil.GetFileName(fullFileName));
                if (System.IO.File.Exists(errorFileName))
                {
                    System.IO.File.Delete(errorFileName);
                }
                System.IO.File.Move(fullFileName, errorFileName);
            }
            catch (Exception ex)
            {
                ArchiveConfigUtil.Instance.PrintLogAction($"移动dii文件到错误文件夹时异常!", true, childServerName, ex);
            }

        }

        #endregion
    }
}
