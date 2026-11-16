using ElincloudServer.CommonBLService;
using ElincloudServer.DBEntity;
using ElincloudServer.DBService;
using ElincloudServer.Entity;

namespace ElincloudServer.ArchiveService
{
    public class ArchiveDatabaseCheckWorker : BackgroundService
    {
        private readonly string childServerName = "归档数据库巡检服务";
        private const int CompleteWaitMinutes = 1;

        public override Task StartAsync(CancellationToken cancellationToken)
        {
            if (ArchiveConfigUtil.Instance.InitSuccess)
            {
                return base.StartAsync(cancellationToken);
            }
            return Task.CompletedTask;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            while (!stoppingToken.IsCancellationRequested)
            {
                await Task.Delay(TimeSpan.FromMinutes(5), stoppingToken);
                try
                {
                    CheckStatusNotChangeExaminations();
                }
                catch (Exception ex)
                {
                    ArchiveConfigUtil.Instance.PrintLogAction("归档数据库巡检异常!", true, childServerName, ex);
                }
            }
        }

        private void CheckStatusNotChangeExaminations()
        {
            List<Hospital> hospitals = DBServices.ElincloudBasicInstance.GetMuch<Hospital>();
            List<string> hospitalGroupIDs = hospitals == null ? new List<string>() : hospitals.Select(p => p.HospitalGroupID).Distinct().ToList();


            foreach (var hospitalGroupID in hospitalGroupIDs)
            {
                DateTime startTime = DateTime.Now.Date;
                DateTime endTime = startTime.AddDays(1);
                SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
                List<RISExamination> risExaminations = instance.GetMuchByProcedure<RISExamination>(ElincloudServer.Common.ApplicationEx.GetStatusNotChangeExamination);
                //超过5分钟的已登记数据改为已检查
                risExaminations = risExaminations.Where(p => (DateTime.Now - p.UpdateDate) > TimeSpan.FromMinutes(5)).ToList();
                if (risExaminations.Count > 0)
                {
                    foreach (var item in risExaminations) {

                        ArchiveConfigUtil.Instance.ArchiveBLInstance.MarkAutoGenerateStudyChecked(hospitalGroupID, item.ImageExaminationID, item.ImagePatientID);
                    }
                }
            }
        }
    }
}
