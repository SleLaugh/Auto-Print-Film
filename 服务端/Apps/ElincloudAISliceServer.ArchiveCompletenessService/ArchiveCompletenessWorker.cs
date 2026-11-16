using ElincloudAISliceServer.DBEntity;
using ElincloudServer.Common;
using ElincloudServer.CommonBLService;
using ElincloudServer.DBService;
using ElincloudServer.Entity;

namespace ElincloudAISliceServer.ArchiveCompletenessService
{
    public class ArchiveCompletenessWorker : BackgroundService
    {
        private readonly ILogger<ArchiveCompletenessWorker> _logger;

        public ArchiveCompletenessWorker(ILogger<ArchiveCompletenessWorker> logger)
        {
            _logger = logger;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            ArchiveCompletenessConfigUtil.Instance.PrintStatusLogAction($"启动成功！");

            while (!stoppingToken.IsCancellationRequested)
            {
                ArchiveCompletenessConfigUtil.Instance.PrintLogAction($"正在运行！");
                try
                {
                    BusinessLogic();
                }
                catch (Exception ex)
                {
                    ArchiveCompletenessConfigUtil.Instance.PrintLogAction($"执行异常!", true, null, ex);
                }
                finally
                {
                    await Task.Delay(5000, stoppingToken);
                }
            }
        }

        private void BusinessLogic()
        {
            ArchiveCompletenessConfigUtil.Instance.PrintLogAction($"共有[{ArchiveCompletenessConfigUtil.Instance.HospitalInfos.Count}]个医院的数据需要处理！");
            foreach (HospitalInfo hospital in ArchiveCompletenessConfigUtil.Instance.HospitalInfos)
            {
                ArchiveCompletenessConfigUtil.Instance.PrintLogAction($"开始处理[{hospital.Name}]的数据！");
                #region 处理单个医院
                SqlAccess instance = DBServices.GetHospitalDBInstance(hospital.HospitalGroupID);

                List<ExaminationImage> list = instance.GetMuch<ExaminationImage>(i =>
                    i.ReceiveCompleted == false
                    && i.CreateDate <= DateTime.Now.AddSeconds(-30)
                );

                if (list != null && list.Count > 0)
                {
                    ArchiveCompletenessConfigUtil.Instance.PrintLogAction($"[{hospital.Name}]需要处理{list.Count()}条数据！");
                    // 每次循环处理的数量
                    int handleCount = 8;
                    // 循环次数
                    int loopCount = (int)Math.Ceiling((double)list.Count / handleCount);
                    // 大循环
                    for (int i = 0; i < loopCount; i++)
                    {
                        // 获取本次循环要处理的数据
                        List<ExaminationImage> sublist = list.Skip(handleCount * i).Take(handleCount).ToList();
                        // 开始本次小循环
                        List<Task> subTasks = new List<Task>();
                        foreach (var item in sublist)
                        {
                            System.Threading.Tasks.Task subtask = new System.Threading.Tasks.Task(new Action(() =>
                            {
                                Process(item, instance);
                            }));
                            subTasks.Add(subtask);
                            subtask.Start();
                        }
                        //等待线程池结束,执行下一次循环
                        Task.WaitAll(subTasks.ToArray());
                    }
                    ArchiveCompletenessConfigUtil.Instance.PrintLogAction($"[{hospital.Name}]的数据处理完成！");
                }
                else
                {
                    ArchiveCompletenessConfigUtil.Instance.PrintLogAction($"[{hospital.Name}]没有要处理的数据！");
                }
                #endregion
            }
        }

        private void Process(ExaminationImage item, SqlAccess instance)
        {
            string path = Utils.StringUtil.Combine(ArchiveCompletenessConfigUtil.Instance.FilePath, item.ImageRelaPath);
            //目录不存在：文件尚未归档到此目录或已被清理搬走，不能判定为收图完成
            //（Directory.GetLastWriteTime 对不存在的路径返回 1601 年，不判存在会 100% 误判成已完成）
            //目录的最后写入时间即为最后一个文件移入该目录的时刻，超过 30 秒无新文件即认为该序列收图完成
            if (Directory.Exists(path) && (DateTime.Now - Directory.GetLastWriteTime(path)).TotalSeconds >= 20)
            {
                item.ReceiveCompleted = true;
                item.ReceiveCompletedDate = DateTime.Now;
                item.SeriesImageCount = Directory.GetFiles(path).Length;
                ElincloudServer.Common.Utils.LogUtil.RecordRunLog($"创建：{item.CreateDate.ToString("G")}  路径：{path} 序列:{item.SeriesInstanceUID} 数量:{item.SeriesImageCount} ");
                instance.Update<ExaminationImage>(item, i => i.ImageRelaPath == item.ImageRelaPath, ExaminationImage._ReceiveCompleted, ExaminationImage._ReceiveCompletedDate, ExaminationImage._SeriesImageCount);
            }
        }
    }
}
