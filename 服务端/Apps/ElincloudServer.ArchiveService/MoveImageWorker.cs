using System.Net;
using System.Threading;
using Dicom;
using ElincloudServer.Common;
using ElincloudServer.CommonBLService;
using ElincloudServer.DBEntity;
using ElincloudServer.DBService;
using ElincloudServer.Entity;

namespace ElincloudServer.ArchiveService
{
    public class MoveImageWorker : BackgroundService
    {
        private readonly ILogger<ArchiveDBWorker> _logger;
        private string childServerName = "移动文件服务";

        public MoveImageWorker(ILogger<ArchiveDBWorker> logger)
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
            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    ArchiveConfigUtil.Instance.PrintLogAction($"正在运行！");

                    List<Hospital> hospitals = DBServices.ElincloudBasicInstance.GetMuch<Hospital>();
                    if (hospitals != null && hospitals.Count > 0)
                    {
                        List<string> hospitalGroupIDs = hospitals.Select(p => p.HospitalGroupID).Distinct().ToList();
                        foreach (var hospitalGroupID in hospitalGroupIDs)
                        {
                            ArchiveConfigUtil.Instance.PrintLogAction($"正在处理分组ID为【{hospitalGroupID}】的影像数据！");
                            DBService.SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
                            List<PendingMoveImages> pendingMoveImages = instance.GetMuch<PendingMoveImages>(p => p.Status == (int)CacheStatus.待处理);
                            if (pendingMoveImages != null && pendingMoveImages.Count > 0)
                            {
                                Transaction trans = instance.GetTransaction();
                                foreach (var item in pendingMoveImages)
                                {
                                    //移动图像
                                    string sourceFolder = Common.Utils.StringUtil.Combine(item.SourceArchiveFolderName, item.ImageRelaPath);
                                    string targetFolder = Common.Utils.StringUtil.Combine(item.TargetArchiveFolderName, item.ImageRelaPath);
                                    await MoveFiles(sourceFolder, targetFolder);
                                    //更新影像移动表的状态
                                    item.Status = (int)CacheStatus.已处理;
                                    item.UpdateDate = DateTime.Now;
                                    trans.Update<PendingMoveImages>(item, p => p.ID == item.ID, PendingMoveImages._Status, PendingMoveImages._UpdateDate);
                                    //更新归档路径
                                    ExaminationImage image = new ExaminationImage()
                                    {
                                        StudyInstanceUID = item.StudyInstanceUID,
                                        SeriesInstanceUID = item.SeriesInstanceUID,
                                        ImageRelaPath = item.ImageRelaPath,
                                        ArchiveFolderName = item.TargetArchiveFolderName
                                    };
                                    trans.Update<ExaminationImage>(image, p => p.StudyInstanceUID == image.StudyInstanceUID && p.SeriesInstanceUID == image.SeriesInstanceUID && p.ImageRelaPath == image.ImageRelaPath, ExaminationImage._ArchiveFolderName);
                                }
                                if (trans.Commit())
                                {
                                    ArchiveConfigUtil.Instance.PrintLogAction($"分组ID为【{hospitalGroupID}】的影像数据移动完成，共处理【{pendingMoveImages.Count}】个序列！");
                                }
                                else
                                {
                                    ArchiveConfigUtil.Instance.PrintLogAction($"分组ID为【{hospitalGroupID}】的影像数据移动完成，共处理【{pendingMoveImages.Count}】个序列，但更新状态出现异常！");
                                }
                            }
                            else
                            {
                                ArchiveConfigUtil.Instance.PrintLogAction($"分组ID为【{hospitalGroupID}】的影像数据移动完成,共处理0个序列！");
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    ArchiveConfigUtil.Instance.PrintLogAction($"执行异常!", true, childServerName, ex);
                }
                finally
                {
                    await Task.Delay(5000, stoppingToken);
                }
            }
        }

        private async Task MoveFiles(string sourceFolder, string targetFolder)
        {
            if (Directory.Exists(sourceFolder))
            {
                if (!Directory.Exists(targetFolder))
                {
                    Directory.CreateDirectory(targetFolder);
                }
                int handleCount = 10;// 控制并发数，避免内存爆炸
                int filesPerThread = 6;//每次处理6个
                SemaphoreSlim semaphore = new SemaphoreSlim(handleCount); //限制并发访问的线程数
                List<string> files = System.IO.Directory.GetFiles(sourceFolder).ToList();
                if (files.Count > 0)
                {
                    List<Task> tasks = new List<Task>();
                    for (int i = 0; i < files.Count; i += filesPerThread)
                    {
                        var subFiles = files.Skip(i).Take(filesPerThread);
                        await semaphore.WaitAsync(); // 控制并发

                        tasks.Add(Task.Run(() =>
                        {
                            foreach (var file in subFiles)
                            {
                                string fileName = Common.Utils.StringUtil.GetFileName(file);
                                string targetFileFullName = Common.Utils.StringUtil.Combine(targetFolder, fileName);
                                if (File.Exists(targetFileFullName))
                                {
                                    File.Delete(file);
                                }
                                else
                                {
                                    File.Move(file, targetFileFullName);
                                }
                            }
                            semaphore.Release(); // 释放信号量
                        }));
                    }
                    await Task.WhenAll(tasks);
                    //删除空文件夹
                    files = System.IO.Directory.GetFiles(sourceFolder).ToList();
                    if (!files.Any())
                    {
                        Common.Utils.FileUtil.DeleteFolder(sourceFolder);
                    }
                }
            }
        }
    }
}
