using ElincloudServer.CommonBLService;
using ElincloudServer.DBEntity;

namespace ElincloudServer.ArchiveService
{
    public class ArchiveCompletenessWorker : BackgroundService
    {
        private readonly string childServerName = "归档完整性检查服务";
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
                await Task.Delay(TimeSpan.FromSeconds(10), stoppingToken);
                try
                {
                    CheckCompletedExaminations();
                }
                catch (Exception ex)
                {
                    ArchiveConfigUtil.Instance.PrintLogAction("检查归档完成状态异常!", true, childServerName, ex);
                }
            }
        }

        private void CheckCompletedExaminations()
        {
            List<KeyValuePair<string, DateTime>> expired = ArchiveConfigUtil.GetExpiredArchiveFileTimes(TimeSpan.FromMinutes(CompleteWaitMinutes));
            if (expired.Count == 0)
            {
                return;
            }

            List<Hospital> hospitals = DBServices.ElincloudBasicInstance.GetMuch<Hospital>();
            List<string> hospitalGroupIDs = hospitals == null ? new List<string>() : hospitals.Select(p => p.HospitalGroupID).Distinct().ToList();

            foreach (var item in expired)
            {
                string[] parts = item.Key.Split(new[] { '|' }, 2);
                if (parts.Length != 2)
                {
                    continue;
                }

                string examinationID = parts[0];
                string patientID = parts[1];

                foreach (var hospitalGroupID in hospitalGroupIDs)
                {
                    ArchiveConfigUtil.Instance.ArchiveBLInstance.MarkAutoGenerateStudyChecked(hospitalGroupID, examinationID, patientID);
                }

                ArchiveConfigUtil.RemoveArchiveFileTime(item.Key);
                ArchiveConfigUtil.Instance.PrintLogAction($"归档完成，检查号[{examinationID}]影像号[{patientID}]状态已更新。", false, childServerName);
            }
        }
    }
}
