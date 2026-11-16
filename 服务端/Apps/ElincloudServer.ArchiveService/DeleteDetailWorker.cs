using ElincloudServer.CommonBLService;
using ElincloudServer.DBEntity;
using ElincloudServer.DBService;
using ElincloudServer.Entity;

namespace ElincloudServer.ArchiveService
{
    public class DeleteDetailWorker : BackgroundService
    {
        private readonly ILogger<ArchiveDBWorker> _logger;
        private string childServerName = "删除归档明细数据服务";

        public DeleteDetailWorker(ILogger<ArchiveDBWorker> logger)
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

        private DateTime lastDeleteDateTime = new DateTime();

        private bool needCalc()
        {
            //每天指定时间进行删除明细数据
            if (DateTime.Now.Hour >= 2)
            {
                //每天删除一次，每天在设置的小时内只触发一次删除
                if (DateTime.Now.Day != lastDeleteDateTime.Day)
                {
                    return true;
                }
            }
            return false;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    ArchiveConfigUtil.Instance.PrintLogAction($"正在运行！");
                    if (needCalc())
                    {
                        List<Hospital> hospitals = DBServices.ElincloudBasicInstance.GetMuch<Hospital>();
                        List<string> hospitalGroupIDs = hospitals.Select(p => p.HospitalGroupID).Distinct().ToList();
                        foreach (var hospitalGroupID in hospitalGroupIDs)
                        {
                            ArchiveConfigUtil.Instance.PrintLogAction($"准备删除数据库【{hospitalGroupID}】下的归档明细数据！");
                            DBService.SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
                            Transaction trans = instance.GetTransaction();
                            bool hasDataToDelete = false;
                            List<Hospital> subHospitals = hospitals.FindAll(p=>p.HospitalGroupID==hospitalGroupID).ToList();
                            foreach (var subHospital in subHospitals)
                            {
                                ArchiveConfigInfo ArchiveConfigInfo = ArchiveConfigUtil.Instance.GetArchiveConfig(subHospital.Code);
                                if (ArchiveConfigInfo!=null&&ArchiveConfigInfo.IsEnableDeleteDetail)
                                {
                                    DateTime dt= DateTime.Now.Date.AddDays(-ArchiveConfigInfo.DetailSaveDays);
                                    int count = instance.GetCount<ExaminationImageDetail>(p => p.CreateDate < dt && p.HospitalNumber == subHospital.Code);
                                    if(count>0)
                                    { 
                                        trans.Delete<ExaminationImageDetail>(p=>p.CreateDate<dt&&p.HospitalNumber==subHospital.Code);
                                        hasDataToDelete = true;
                                    }
                                }
                            }
                            if (hasDataToDelete)
                            {
                                if (trans.Commit())
                                {
                                    ArchiveConfigUtil.Instance.PrintLogAction($"已成功删除数据库【{hospitalGroupID}】下的归档明细数据！");
                                }
                                else
                                {
                                    ArchiveConfigUtil.Instance.PrintLogAction($"删除数据库【{hospitalGroupID}】下的归档明细数据出现异常！");
                                }
                            }
                            else
                            {
                                ArchiveConfigUtil.Instance.PrintLogAction($"数据库【{hospitalGroupID}】没有需要删除的数据！");
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
                    await Task.Delay(1000*60*5, stoppingToken);
                }
            }
        }
    }
}
