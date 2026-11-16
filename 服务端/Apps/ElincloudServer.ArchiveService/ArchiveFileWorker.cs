using System.Data;
using Dicom;
using Dicom.Imaging;
using ElincloudServer.Common;
using ElincloudServer.Entity;
using static Org.BouncyCastle.Math.EC.ECCurve;
using DCMImageInfo = ElincloudAISliceServer.Entity.DCMImageInfo;

namespace ElincloudServer.ArchiveService
{
    public class ArchiveFileWorker : BackgroundService
    {
        private ArchiveConfigUtil archiveConfigUtilInstance = null;
        private string childServerName = "文件处理服务";
        private ArchiveUtil archiveUtil = null;
        public ArchiveFileWorker(int instanceID)
        {
            archiveUtil = new ArchiveUtil();
            archiveConfigUtilInstance = new ArchiveConfigUtil();
            archiveConfigUtilInstance.Init(instanceID);
            childServerName = childServerName + instanceID.ToString();
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
            if (System.IO.Directory.Exists(archiveConfigUtilInstance.ImageSourcePath))
            {
                if (System.IO.Directory.Exists(archiveConfigUtilInstance.ImageTargetPath))
                {
                    string ArchiveFolderCacheFileName = Common.Utils.StringUtil.Combine(AppDomain.CurrentDomain.BaseDirectory, "archiveFolderCache.txt");
                    archiveFolderCacheManager = new ArchiveFolderCacheManager(ArchiveFolderCacheFileName);//启动缓存

                    archiveConfigUtilInstance.PrintStatusLogAction($"启动成功！归档数据目录[{archiveConfigUtilInstance.ImageSourcePath}]!", childServerName);
                    int handleCount = 100;//一批处理100个文件
                    while (!stoppingToken.IsCancellationRequested)
                    {
                        try
                        {
                            archiveConfigUtilInstance.PrintLogAction($"正在运行！", false, childServerName);
                            //每归档完成一批文件都进行一次垃圾回收
                            GC.Collect();

                            #region 对获取的文件进行排序  日期早的先进行归档

                            DirectoryInfo dir = new DirectoryInfo(archiveConfigUtilInstance.ImageSourcePath);
                            List<string> files = new List<string>();
                            var fsInfos = dir.GetFileSystemInfos().Where(p => (DateTime.Now - p.CreationTime).TotalSeconds > 2 && p is FileInfo);
                            if (fsInfos != null && fsInfos.Count() > 0)
                            {
                                fsInfos = fsInfos.OrderBy(p => p.CreationTime);
                                files = fsInfos.Select(p => p.FullName).ToList();
                            }

                            #endregion

                            if (files.Count > 0)
                            {
                                archiveConfigUtilInstance.PrintLogAction($"开始归档...文件总数: {files.Count} 个", false, childServerName);
                                System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
                                sw.Start();
                                int loopCount = (int)Math.Ceiling((double)files.Count / handleCount);
                                int remainingTotal = files.Count;
                                for (int i = 0; i < loopCount; i++)
                                {

                                    List<string> subFiles = files.Skip(handleCount * i).Take(handleCount).ToList();
                                    foreach (var file in subFiles)
                                    {
                                        try
                                        {
                                            ArchiveFile(file);
                                        }
                                        catch (Exception ex)
                                        {
                                            archiveConfigUtilInstance.PrintLogAction($"归档文件[{file}]异常!", true, childServerName, ex);
                                        }
                                    }
                                    remainingTotal = remainingTotal - subFiles.Count;
                                    archiveConfigUtilInstance.PrintLogAction(string.Format("[{0}]正在归档...运行状态: {1}/{2}", archiveConfigUtilInstance.ImageSourcePath, remainingTotal, files.Count), false, childServerName);

                                    //一个循环组结束统一记录到一个文件中
                                    string json = ElincloudServer.Common.Utils.JsonUtil.ToJson(archiveConfigUtilInstance.ArchiveDetails);
                                    string fileFullName = ElincloudServer.Common.Utils.StringUtil.Combine(archiveConfigUtilInstance.ImageInfoPath, DateTime.Now.ToString("yyyyMMddHHmmssffff") + Guid.NewGuid().ToString("N").Substring(0, 10) + ".dii");
                                    System.IO.File.WriteAllText(fileFullName, json);
                                    archiveConfigUtilInstance.ArchiveDetails.Clear();
                                }
                                sw.Stop();
                                Console.WriteLine("归档完成...耗时" + sw.ElapsedMilliseconds + "ms");
                                archiveConfigUtilInstance.PrintLogAction($"监听归档文件夹...空闲", false, childServerName);
                            }
                            else
                            {
                                archiveConfigUtilInstance.PrintLogAction($"监听归档文件夹...空闲", false, childServerName);
                                await Task.Delay(1000, stoppingToken);
                            }
                        }
                        catch (Exception ex)
                        {
                            archiveConfigUtilInstance.PrintLogAction($"监听归档文件夹异常!", true, childServerName, ex);
                        }
                    }
                }
                else
                {
                    if (archiveConfigUtilInstance.ImageTargetPath == string.Empty)
                    {
                        archiveConfigUtilInstance.PrintLogAction($"启动失败！归档目标文件夹路径未配置!", true, childServerName);
                    }
                    else
                    {
                        archiveConfigUtilInstance.PrintLogAction($"启动失败！归档目标文件夹路径[{archiveConfigUtilInstance.ImageSourcePath}]不存在!", true, childServerName);
                    }

                }
            }
            else
            {
                if (archiveConfigUtilInstance.ImageSourcePath == string.Empty)
                {
                    archiveConfigUtilInstance.PrintLogAction($"启动失败！归档文件夹路径未配置!", true, childServerName);
                }
                else
                {
                    archiveConfigUtilInstance.PrintLogAction($"启动失败！归档文件夹路径[{archiveConfigUtilInstance.ImageSourcePath}]不存在!", true, childServerName);
                }
            }
        }

        #region 归档操作

        private void ArchiveFile(string sourceFileFullName)
        {
            try
            {
                if (System.IO.File.Exists(sourceFileFullName))
                {
                    DCMModel model = archiveUtil.GetModel(sourceFileFullName);
                    if (model != null)
                    {
                        ArchiveConfigInfo archiveConfig = archiveConfigUtilInstance.GetArchiveConfig(model.HospitalNumber);
                        if (archiveConfig != null)
                        {
                            string archiveFolderName = GetArchiveFolderName(model);
                            if (string.IsNullOrEmpty(archiveFolderName))
                            {
                                string contentMsg = "归档目录";
                                archiveConfigUtilInstance.PrintLogAction($"没有获取到归档配置信息中的{contentMsg}!", false, childServerName);
                            }
                            else
                            {
                                string targetArchivePath = ElincloudServer.Common.Utils.StringUtil.Combine(archiveFolderName, archiveConfigUtilInstance.ImageFolder);
                                ClientResult ret = archiveUtil.GetArchiveTargetFileName(model, sourceFileFullName, targetArchivePath, archiveConfig);
                                if (ret.Success)
                                {
                                    string targetFileFullName = ret.Result.ToString();
                                    if (model.NumberOfFrames == "" || model.NumberOfFrames == "1")//单帧图像
                                    {
                                        ClientResult saveRet = archiveUtil.SaveDcmFileAndGetDcmImageInfo(sourceFileFullName, targetFileFullName, archiveFolderName, model);
                                        if (saveRet.Success)
                                        {
                                            DCMImageInfo info = saveRet.Result as DCMImageInfo;
                                            if (archiveConfigUtilInstance.IsEnableMultArchiveFolder)//启用多路径归档时将实际归档的目录进行存储
                                            {
                                                info.ArchiveFolderName = archiveFolderName;
                                            }
                                            archiveConfigUtilInstance.ArchiveDetails.Add(info.HospitalNumber, info);
                                            ArchiveConfigUtil.UpdateArchiveFileTime(info.ExaminationID, info.PatientID, DateTime.Now);
                                        }
                                        else
                                        {
                                            archiveConfigUtilInstance.PrintLogAction(saveRet.Message, false, childServerName);
                                            ArchiveFailed(sourceFileFullName);
                                        }
                                    }
                                    else//多帧图像，这里在归档的时候把图像中的每一帧单独归档存储，保证后期在影像调阅的时候的调图速度
                                    {
                                        int frames = 1;
                                        bool success = int.TryParse(model.NumberOfFrames, out frames);
                                        if (success)
                                        {
                                            string sourceFolder = Utils.StringUtil.GetDirectoryName(sourceFileFullName);
                                            string targetFolder = Utils.StringUtil.GetDirectoryName(targetFileFullName);
                                            Dictionary<string, DCMModel> dic = getMultiFramesDicomFileArchivePath(sourceFileFullName, targetFileFullName, model, frames);
                                            foreach (var kv in dic)
                                            {
                                                string sourceFrameFileFullName = ElincloudServer.Common.Utils.StringUtil.Combine(sourceFolder, kv.Key);
                                                string targetFrameFileFullName = ElincloudServer.Common.Utils.StringUtil.Combine(targetFolder, kv.Key);
                                                ClientResult saveRet = archiveUtil.SaveDcmFileAndGetDcmImageInfo(sourceFrameFileFullName, targetFrameFileFullName, archiveFolderName, kv.Value);
                                                if (saveRet.Success)
                                                {
                                                    DCMImageInfo info = saveRet.Result as DCMImageInfo;
                                                    if (archiveConfigUtilInstance.IsEnableMultArchiveFolder)//启用多路径归档时将实际归档的目录进行存储
                                                    {
                                                        info.ArchiveFolderName = archiveFolderName;
                                                    }
                                                    archiveConfigUtilInstance.ArchiveDetails.Add(info.HospitalNumber, info);
                                                    ArchiveConfigUtil.UpdateArchiveFileTime(info.ExaminationID, info.PatientID, DateTime.Now);
                                                }
                                                else
                                                {
                                                    archiveConfigUtilInstance.PrintLogAction(saveRet.Message, false, childServerName);
                                                    ArchiveFailed(sourceFileFullName);
                                                }
                                            }
                                            System.IO.File.Delete(sourceFileFullName);
                                        }
                                    }

                                }
                                else
                                {
                                    archiveConfigUtilInstance.PrintLogAction(ret.Message, false, childServerName);
                                    ArchiveFailed(sourceFileFullName);
                                }
                            }
                        }
                        else
                        {
                            string contentMsg = "归档配置信息";
                            archiveConfigUtilInstance.PrintLogAction($"没有读取到归档文件[{sourceFileFullName}]的{contentMsg}!", false, childServerName);
                            ArchiveFailed(sourceFileFullName);
                        }
                    }
                    else
                    {
                        string contentMsg = "内容";
                        archiveConfigUtilInstance.PrintLogAction($"没有读取到归档文件[{sourceFileFullName}]的{contentMsg}!", false, childServerName);
                        ArchiveFailed(sourceFileFullName);
                    }

                }
                else
                {
                    archiveConfigUtilInstance.PrintLogAction($"要归档的文件[{sourceFileFullName}]不存在!", false, childServerName);
                }
            }
            catch (Exception ex)
            {
                if (System.IO.File.Exists(sourceFileFullName))
                {
                    ArchiveFailed(sourceFileFullName);
                }
                archiveConfigUtilInstance.PrintLogAction($"归档文件[{sourceFileFullName}]出现异常!", true, childServerName, ex);
            }

        }

        /// <summary>
        /// 拆分多帧图像
        /// </summary>
        /// <param name="sourceFileFullName">原始多帧图像文件路径</param>
        /// <param name="targetFileFullName">原始多帧图像归档路径</param>
        /// <param name="model">原始多帧图像归档信息</param>
        /// <param name="numberOfFrames">原始多帧图像帧数</param>
        /// <returns>Key：拆分后单帧图像文件名，Value：拆分后单帧图像归档信息</returns>
        private Dictionary<string, DCMModel> getMultiFramesDicomFileArchivePath(string sourceFileFullName, string targetFileFullName, DCMModel model, int numberOfFrames)
        {
            Dictionary<string, DCMModel> dic = new Dictionary<string, DCMModel>();
            DicomFile file = DicomFile.Open(sourceFileFullName);
            string sourceUID = file.Dataset.GetSingleValueOrDefault<string>(DicomTag.SOPInstanceUID, string.Empty);
            DicomPixelData sourcePixelData = DicomPixelData.Create(file.Dataset);
            //string targetFolder = Utils.StringUtil.GetDirectoryName(targetFileFullName);
            string targetFileNameWithoutExt = Utils.StringUtil.GetFileNameWithoutExtension(targetFileFullName);
            string targetFileExt = System.IO.Path.GetExtension(targetFileFullName);
            bool IsUpdateUID = (sourceUID + "." + numberOfFrames.ToString()).Length > 64;
            for (int frame = 1; frame <= numberOfFrames; frame++)
            {
                DCMModel frameModel = model.Clone();

                //1.复制dicom文件形成单帧文件，并清空PixelData，后面将写入单帧图像
                DicomFile frameFile = file.Clone();
                //frameFile.Dataset.Remove(DicomTag.PixelData); // 移除PixelData标签以节省内存
                KeepCurrentMultiFrameInfo(frameFile.Dataset, frame, numberOfFrames);
                //2.重写单帧文件的图像序号
                frameFile.Dataset.AddOrUpdate<int>(DicomTag.InstanceNumber, frame);
                frameModel.InstanceNumber = frame.ToString();
                //3.重写单帧文件的InstanceUID
                string frameUID = sourceUID + "." + frame;
                //将最后一个.后面的字符截取掉，原始是SOPInstanceUID的限制为64个字符，有些图像中的值已经达到64
                if (IsUpdateUID)
                {
                    frameUID = sourceUID.Substring(0, sourceUID.Length - 5).Trim('.') + "." + frame;
                }
                frameFile.Dataset.AddOrUpdate<string>(DicomTag.SOPInstanceUID, frameUID);
                frameModel.SOPInstanceUID = frameUID;
                //4.重写单帧文件的NumberOfFrames
                //Dicom文件中的tag值不用改，默认为1
                frameModel.NumberOfFrames = frame.ToString();//model里面图像帧数更新为1
                //5.重写单帧文件的图像信息
                var framePixelData = sourcePixelData.GetFrame(frame - 1);
                DicomPixelData frameDicomPixelData = DicomPixelData.Create(frameFile.Dataset, true);
                frameDicomPixelData.AddFrame(framePixelData);
                //6.将单帧文件保存到归档目录
                string frameFileName = targetFileNameWithoutExt + "." + frame + targetFileExt;
                string frameSourceFileFullName = ElincloudServer.Common.Utils.StringUtil.Combine(archiveConfigUtilInstance.ImageSourcePath, frameFileName);
                //string frameTargetFileFullName = ElincloudServer.Common.Utils.StringUtil.Combine(targetFolder, frameFileName);
                frameFile.Save(frameSourceFileFullName);
                //7.写入返回值集合，从外面移动文件
                dic.Add(frameFileName, frameModel);
            }
            return dic;
        }

        /// <summary>
        /// 保留当前帧的信息
        /// </summary>
        /// <param name="dataset"></param>
        /// <param name="frameIndex"></param>
        /// <param name="totalFrames"></param>
        private void KeepCurrentMultiFrameInfo(DicomDataset dataset, int frameIndex, int totalFrames)
        {
            //只保留当前帧的信息
            if (dataset.TryGetSequence(DicomTag.PerFrameFunctionalGroupsSequence, out var perFrameSeq) &&
                frameIndex - 1 < perFrameSeq.Items.Count)
            {
                var currentFrame = perFrameSeq.Items[frameIndex - 1];

                ExtractAllSequencesToRoot(dataset, currentFrame);

                dataset.Remove(DicomTag.PerFrameFunctionalGroupsSequence);
            }
        }
        /// <summary>
        /// 提取所有帧的tag到根节点
        /// </summary>
        /// <param name="rootDataset"></param>
        /// <param name="sourceDataset"></param>
        private void ExtractAllSequencesToRoot(DicomDataset rootDataset, DicomDataset sourceDataset)
        {
            foreach (var item in sourceDataset)
            {
                if (item.ValueRepresentation == DicomVR.SQ)
                {
                    // 处理序列
                    var seq = sourceDataset.GetSequence(item.Tag);
                    if (seq.Items.Count > 0)
                    {
                        // 将序列中的第一项（当前帧的数据）添加到根数据集
                        rootDataset.AddOrUpdate(seq.Items[0]);
                    }
                }
                else
                {
                    // 直接复制非序列项
                    rootDataset.AddOrUpdate(item);
                }
            }
        }


        private void ArchiveFailed(string fullFileName)
        {
            try
            {
                string errorFileName = Utils.StringUtil.Combine(archiveConfigUtilInstance.ImageSourceErrorPath, Utils.StringUtil.GetFileName(fullFileName));
                if (System.IO.File.Exists(errorFileName))
                {
                    System.IO.File.Delete(errorFileName);
                }
                System.IO.File.Move(fullFileName, errorFileName);
            }
            catch (Exception ex)
            {
                archiveConfigUtilInstance.PrintLogAction("移动dicom文件到错误文件夹时异常！", true, childServerName, ex);
            }
        }

        #region 归档路径目录

        private ArchiveFolderCacheManager archiveFolderCacheManager = null;

        DateTime DiskSpaceTestDt = DateTime.Now.AddMinutes(-10);
        private string LastArchiveFolderName = string.Empty;

        private string GetArchiveFolderName(DCMModel model)
        {
            try
            {
                if (archiveConfigUtilInstance.IsEnableMultArchiveFolder)
                {
                    string patientID = string.IsNullOrWhiteSpace(model.PatientID) ? model.AccessionNumber : model.PatientID;
                    ArchiveFolderCache archiveFolderCache = archiveFolderCacheManager.GetByPatientId(patientID);
                    if (archiveFolderCache != null)
                    {
                        return archiveFolderCache.ArchiveFolderName;
                    }
                    else
                    {
                        string ArchiveFolderName = string.Empty;
                        if ((DateTime.Now - DiskSpaceTestDt).TotalMinutes > 5)//5分钟验证下磁盘空间，并返回可存储的归档路径
                        {
                            foreach (var item in archiveConfigUtilInstance.ArchiveFolderConfigInfos)
                            {
                                if (!string.IsNullOrWhiteSpace(item.FolderName))
                                {
                                    string inputPath = item.FolderName.Trim();
                                    string rootPath = Path.GetPathRoot(inputPath);
                                    if (!string.IsNullOrEmpty(rootPath))
                                    {
                                        if (!rootPath.EndsWith(Path.DirectorySeparatorChar.ToString()))
                                        {
                                            rootPath += Path.DirectorySeparatorChar;
                                        }
                                        var drive = new DriveInfo(rootPath);
                                        double freeGB = drive.AvailableFreeSpace / 1024.0 / 1024 / 1024;
                                        if (freeGB >= item.DiskSpaceLimit)
                                        {
                                            ArchiveFolderName = item.FolderName;
                                            break;
                                        }
                                    }
                                }
                            }
                            LastArchiveFolderName = ArchiveFolderName;
                            archiveFolderCache = new ArchiveFolderCache()
                            {
                                ArchiveFolderName = ArchiveFolderName,
                                PatientID = patientID,
                                CacheDate = DateTime.Now,
                            };
                            archiveFolderCacheManager.Add(archiveFolderCache);
                            return ArchiveFolderName;
                        }
                        else
                        {
                            return LastArchiveFolderName;
                        }
                    }
                }
                else
                {
                    return archiveConfigUtilInstance.ImageTargetPath;
                }
            }
            catch (Exception ex)
            {
                Utils.LogUtil.RecordErrorLog("获取归档目录异常", ex);
            }
            //return archiveConfigUtilInstance.ImageTargetPath;
            return string.Empty;
        }

        #endregion

        #endregion

    }
}
