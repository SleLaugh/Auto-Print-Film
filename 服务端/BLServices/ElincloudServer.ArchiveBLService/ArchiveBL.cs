using ElincloudAISliceServer.Entity;
using ElincloudServer.Common;
using ElincloudServer.CommonBLService;
using ElincloudServer.DBEntity;
using ElincloudServer.DBService;
using ElincloudServer.Entity;
using System.Data;
using System.Linq.Expressions;
using System.Reflection;
using DCMImageInfo = ElincloudAISliceServer.Entity.DCMImageInfo;
using ExaminationImage = ElincloudAISliceServer.DBEntity.ExaminationImage;
using RISExamination = ElincloudAISliceServer.DBEntity.RISExamination;

namespace ElincloudServer.ArchiveBLService
{
    public class ArchiveBL
    {
        private static MultiDictionary<string, ExaminationImage> imageDics = new MultiDictionary<string, ExaminationImage>();
        private static MultiDictionary<string, ExaminationImageDetail> detailDics = new MultiDictionary<string, ExaminationImageDetail>();

        public void FinishArchive(DCMImageInfo archiveInfo, ArchiveConfigInfo archiveConfig, CheckInfo studyType)
        {
            try
            {
                if (archiveInfo != null)
                {
                    SqlAccess instance = DBServices.GetHospitalDBInstance(archiveConfig.Hospital.HospitalGroupID);
                    Transaction trans = null;
                    if (!archiveConfig.IsEnableArchiveCache)
                    {
                        trans = instance.GetTransaction();
                    }
                    DateTime studyDate;
                    if (!DateTime.TryParse(archiveInfo.StudyDate + " " + archiveInfo.StudyTime, out studyDate))
                    {
                        studyDate = DateTime.Now;
                    }
                    if (archiveConfig.IsModifyExaminationStatus)
                    {
                        if (studyType != null)
                        {
                            if (studyType.SysType == SysTypes.RIS)
                            {
                                #region 放射系统修改检查状态

                                RISExamination study = null;

                                if (archiveInfo.PatientID == archiveInfo.ExaminationID)
                                {
                                    #region 解决当配置的检查类型传过来的图像没有检查号的时候，通过影像号和检查类型自动匹配检查号

                                    List<string> matchModalityList = archiveConfig.AutoMatchExaminationIDStudyTypes.FindAll(p => p.SysType == studyType.SysType).Select(p => p.Name).ToList();
                                    //如果检查号和影像号一样 说明现场图像里面没有检查号,归档程序自动将影像号给到了检查号字段中
                                    //此处if中从examination表中获取真正的检查号
                                    if (matchModalityList.Count > 0)
                                    {
                                        RISExamination lastEx = null;
                                        List<RISExamination> vs = new List<RISExamination>();
                                        if (matchModalityList.Any(p => p == archiveInfo.StudyTypeName))
                                        {
                                            //当图像中的检查类型是正确的时候根据图像检查类型精准匹配
                                            vs = instance.GetMuch<RISExamination>(p => p.ImagePatientID == archiveInfo.PatientID && p.StudyTypeName == archiveInfo.StudyTypeName);
                                        }
                                        else
                                        {
                                            /*当图像中的检查类型也是错误的情况下只能从配置的检查类型中模糊匹配，会出现匹配错误的情况,情况如下
                                             *当一个系统类型下如CT和DR两个台设备发过来的图都没有检查号且Modality是错的情况下，有一个患者
                                             *先登记了DR后登记了CT，然后患者去做DR检查，检查完成后就会自动匹配成CT的那个检查号, 因为下面是按检查日期进行降序排序后取了第一条数据*/
                                            vs = instance.GetMuch<RISExamination>(p => p.ImagePatientID == archiveInfo.PatientID && matchModalityList.Contains(p.StudyTypeName));
                                        }
                                        if (vs.Count > 0)
                                        {
                                            var orderedList = vs.OrderByDescending(p => p.StudyDate);
                                            var first = orderedList.FirstOrDefault();
                                            lastEx = instance.GetOne<RISExamination>(p => p.ID == first.ID);
                                        }


                                        if (lastEx != null)
                                        {
                                            archiveInfo.ExaminationID = lastEx.ID;
                                            study = lastEx;
                                        }
                                    }

                                    #endregion
                                }
                                else
                                {
                                    study = instance.GetOne<RISExamination>(p => p.ImagePatientID == archiveInfo.PatientID && p.ImageExaminationID == archiveInfo.ExaminationID);
                                }


                                if (study != null)
                                {
                                    //有任何一个序列传上来的时候检查状态变为已检查
                                    if (false && study.Status == (int)ExaminationStatus.已登记)
                                    {
                                        study.Status = (int)ExaminationStatus.已检查;
                                        study.StudyDate = studyDate;
                                        study.UpdateDate = DateTime.Now;
                                        trans.Update<RISExamination>(study, p => p.ID == study.ID, RISExamination._Status, RISExamination._StudyDate, RISExamination._UpdateDate);
                                    }
                                }

                                #endregion
                            }
                            else if (studyType.SysType == SysTypes.ECG)
                            {
                                #region 心电系统修改检查状态

                                ECGExamination study = null;

                                if (archiveInfo.PatientID == archiveInfo.ExaminationID)
                                {
                                    #region 解决当配置的检查类型传过来的图像没有检查号的时候，通过影像号和检查类型自动匹配检查号

                                    List<string> matchModalityList = archiveConfig.AutoMatchExaminationIDStudyTypes.FindAll(p => p.SysType == studyType.SysType).Select(p => p.Name).ToList();
                                    //如果检查号和影像号一样 说明现场图像里面没有检查号,归档程序自动将影像号给到了检查号字段中
                                    //此处if中从examination表中获取真正的检查号
                                    if (matchModalityList.Count > 0)
                                    {
                                        ECGExamination lastEx = null;
                                        List<ECGExamination> vs = new List<ECGExamination>();
                                        if (matchModalityList.Any(p => p == archiveInfo.StudyTypeName))
                                        {
                                            //当图像中的检查类型是正确的时候根据图像检查类型精准匹配
                                            vs = instance.GetMuch<ECGExamination>(p => p.ImagePatientID == archiveInfo.PatientID && p.StudyTypeName == archiveInfo.StudyTypeName);
                                        }
                                        else
                                        {
                                            /*当图像中的检查类型也是错误的情况下只能从配置的检查类型中模糊匹配，会出现匹配错误的情况,情况如下
                                             *当一个系统类型下如CT和DR两个台设备发过来的图都没有检查号且Modality是错的情况下，有一个患者
                                             *先登记了DR后登记了CT，然后患者去做DR检查，检查完成后就会自动匹配成CT的那个检查号, 因为下面是按检查日期进行降序排序后取了第一条数据*/
                                            vs = instance.GetMuch<ECGExamination>(p => p.ImagePatientID == archiveInfo.PatientID && matchModalityList.Contains(p.StudyTypeName));
                                        }
                                        if (vs.Count > 0)
                                        {
                                            var orderedList = vs.OrderByDescending(p => p.StudyDate);
                                            var first = orderedList.FirstOrDefault();
                                            lastEx = instance.GetOne<ECGExamination>(p => p.ID == first.ID);
                                        }


                                        if (lastEx != null)
                                        {
                                            archiveInfo.ExaminationID = lastEx.ID;
                                            study = lastEx;
                                        }
                                    }

                                    #endregion
                                }
                                else
                                {
                                    study = instance.GetOne<ECGExamination>(p => p.ImagePatientID == archiveInfo.PatientID && p.ImageExaminationID == archiveInfo.ExaminationID);
                                }


                                if (study != null)
                                {
                                    //有任何一个序列传上来的时候检查状态变为已检查
                                    if (false && study.Status == (int)ExaminationStatus.已登记)
                                    {
                                        study.Status = (int)ExaminationStatus.已检查;
                                        study.StudyDate = studyDate;
                                        study.UpdateDate = DateTime.Now;
                                        trans.Update<ECGExamination>(study, p => p.ID == study.ID, ECGExamination._Status, ECGExamination._StudyDate, ECGExamination._UpdateDate);
                                    }
                                }

                                #endregion
                            }
                        }
                    }


                    #region 添加或更新ExaminationImage

                    DataStoreConfigInfo configInfo = CommonBL.GetDataStoreConfigInfo();
                    DateTime ArchiveDate = DateTime.Now;
                    if (configInfo.HospitalSavedDaysConfigInfos != null && configInfo.HospitalSavedDaysConfigInfos.Count > 0)
                    {
                        DataStoreHospitalSavedDaysConfigInfo savedDaysConfigInfo = configInfo.HospitalSavedDaysConfigInfos.FirstOrDefault(p => p.Category == DicomImageCategories.原始影像 && p.HospitalID == archiveConfig.Hospital.ID);
                        if (savedDaysConfigInfo != null)
                        {
                            if ((ArchiveDate - studyDate).TotalDays > savedDaysConfigInfo.SavedDays)
                            {
                                int days = savedDaysConfigInfo.SavedDays > 3 ? savedDaysConfigInfo.SavedDays - 3 : 0;
                                ArchiveDate = DateTime.Now.AddDays(-days);
                            }
                        }
                    }

                    ExaminationImage image = new ExaminationImage()
                    {
                        PatientID = archiveInfo.PatientID,
                        ExaminationID = archiveInfo.ExaminationID,
                        StudyInstanceUID = archiveInfo.StudyInstanceUID,
                        SeriesInstanceUID = archiveInfo.SeriesInstanceUID,
                        HospitalNumber = archiveInfo.HospitalNumber,
                        StudyTypeName = archiveInfo.StudyTypeName,
                        PatientName = archiveInfo.PatientName,
                        PatientAge = archiveInfo.PatientAge,
                        PatientSex = archiveInfo.PatientSex,
                        PatientBirthday = archiveInfo.PatientBirthday,
                        SeriesNumber = archiveInfo.SeriesNumber,
                        StudyDate = studyDate,
                        ImageRelaPath = archiveInfo.ImageRelaPath,
                        Category = (int)DicomImageCategories.原始影像,
                        IsArchived = (int)DataStoreStatus.未归档,
                        DataStoreType = (int)DataStoreTypes.UnKown,
                        ArchiveDate = ArchiveDate,
                        ArchiveFolderName = archiveInfo.ArchiveFolderName,
                        DiskFileSize = -1,
                        ArchiveFileSize = -1,
                        SeriesImageCount = 0,
                        ScanThickness = archiveInfo.ScanThickness,
                        BodyPartExamined = archiveInfo.BodyPartExamined,
                        SeriesDescription = archiveInfo.SeriesDescription,
                        ReceiveCompleted = false,
                        MachineKey = archiveInfo.MachineKey,
                    };
                    if (archiveConfig.IsEnableArchiveCache)
                    {
                        imageDics.Add(archiveConfig.Hospital.HospitalGroupID, image);
                    }
                    else
                    {
                        ExaminationImage ei = instance.GetOne<ExaminationImage>(p => p.ImageRelaPath == archiveInfo.ImageRelaPath);
                        if (ei == null)
                        {
                            trans.Insert<ExaminationImage>(image);
                        }
                        else
                        {
                            //验证图像是否需要移动，若需要，则插入记录
                            if (!string.IsNullOrWhiteSpace(ei.ArchiveFolderName) && !string.IsNullOrWhiteSpace(archiveInfo.ArchiveFolderName) && ei.ArchiveFolderName != archiveInfo.ArchiveFolderName)
                            {
                                PendingMoveImages pendingMoveImages = new PendingMoveImages()
                                {
                                    ID = instance.GetID<PendingMoveImages>(10),
                                    StudyInstanceUID = ei.StudyInstanceUID,
                                    SeriesInstanceUID = ei.SeriesInstanceUID,
                                    ImageRelaPath = ei.ImageRelaPath,
                                    SourceArchiveFolderName = ei.ArchiveFolderName,
                                    TargetArchiveFolderName = archiveInfo.ArchiveFolderName,
                                    CreateDate = DateTime.Now,
                                    UpdateDate = DateTime.Now,
                                    Status = (int)CacheStatus.待处理
                                };
                                trans.Insert<PendingMoveImages>(pendingMoveImages);
                            }
                            //在启用自动登记的时候，上面Insert方法里面插入到表中的PaitentID和ExaminationID是图像中的信息，在自动登记之后ExaminationImage表中
                            //的影像号和检查号会改为ris生成的号码，如果这些图像过一段时间重新归档后，不用把系统的影像号和检查号更新回图像中的号码，这样自动登记那里也不用重新再把号码该成ris里面的号
                            //ei.PatientID = archiveInfo.PatientID;
                            //ei.ExaminationID = archiveInfo.ExaminationID;
                            ei.StudyInstanceUID = archiveInfo.StudyInstanceUID;
                            ei.SeriesInstanceUID = archiveInfo.SeriesInstanceUID;
                            ei.HospitalNumber = archiveInfo.HospitalNumber;
                            ei.StudyTypeName = archiveInfo.StudyTypeName;
                            ei.PatientName = archiveInfo.PatientName;
                            ei.PatientAge = archiveInfo.PatientAge;
                            ei.PatientSex = archiveInfo.PatientSex;
                            ei.PatientBirthday = archiveInfo.PatientBirthday;
                            ei.SeriesNumber = archiveInfo.SeriesNumber;
                            ei.StudyDate = studyDate;
                            ei.Category = (int)DicomImageCategories.原始影像;
                            //重新归档图像要重置归档状态
                            ei.IsArchived = (int)DataStoreStatus.未归档;
                            ei.DataStoreType = (int)DataStoreTypes.UnKown;
                            ei.DiskFileSize = -1;
                            ei.ArchiveFileSize = -1;
                            ei.SeriesImageCount = 0;
                            ei.ArchiveDate = ArchiveDate;
                            ei.ScanThickness = archiveInfo.ScanThickness;
                            ei.BodyPartExamined = archiveInfo.BodyPartExamined;
                            ei.SeriesDescription = archiveInfo.SeriesDescription;
                            ei.ReceiveCompleted = false;
                            ei.MachineKey = archiveInfo.MachineKey;

                            trans.Update<ExaminationImage>(ei, p => p.ImageRelaPath == archiveInfo.ImageRelaPath,
                                ExaminationImage._StudyInstanceUID,
                                ExaminationImage._SeriesInstanceUID,
                                ExaminationImage._HospitalNumber,
                                ExaminationImage._StudyTypeName,
                                ExaminationImage._PatientName,
                                ExaminationImage._PatientAge,
                                ExaminationImage._PatientSex,
                                ExaminationImage._PatientBirthday,
                                ExaminationImage._SeriesNumber,
                                ExaminationImage._StudyDate,
                                ExaminationImage._Category,
                                ExaminationImage._IsArchived,
                                ExaminationImage._DataStoreType,
                                ExaminationImage._DiskFileSize,
                                ExaminationImage._ArchiveFileSize,
                                ExaminationImage._SeriesImageCount,
                                ExaminationImage._ArchiveDate,
                                ExaminationImage._BodyPartExamined,
                                ExaminationImage._ScanThickness,
                                ExaminationImage._SeriesDescription,
                                ExaminationImage._ReceiveCompleted,
                                ExaminationImage._MachineKey
                                );
                        }
                    }


                    #endregion
                    if (trans != null)
                    {
                        if (!SqlAccess.ExecuteTransactions(trans))
                        {
                            throw new ServiceException("更新归档信息失败！");
                        }
                    }

                }
                else
                {
                    throw new ServiceException("要更新服务的归档信息不能为空!");
                }
            }
            catch (Exception)
            {

            }
        }

        public void AutoGenerateCloudImageReport(DCMImageInfo info, ArchiveConfigInfo archiveConfig, CheckInfo studyType)
        {
            if (info != null)
            {
                SqlAccess instance = DBServices.GetHospitalDBInstance(archiveConfig.Hospital.HospitalGroupID);

                #region 根据图像信息查询云影像报告

                List<string> scpReportKeys = new List<string>(archiveConfig.Keys);
                scpReportKeys.Add(CloudImageReport._ExaminationID);//通过图像直接生成云影像信息的场景下，检查与报告必须是一对一关系，后续第三方提供报告视图的时候ReportID不能给值
                System.Linq.Expressions.Expression<Func<CloudImageReport, bool>> whereExpressionReport = null;
                ParameterExpression parameterReport = Expression.Parameter(typeof(CloudImageReport), "p");
                foreach (var key in scpReportKeys)
                {
                    MemberExpression member = Expression.PropertyOrField(parameterReport, key);
                    ConstantExpression constant = Expression.Constant(Convert.ChangeType(info.GetType().GetProperty(key).GetValue(info, null), member.Type));
                    BinaryExpression query = Expression.Equal(member, constant);
                    if (query != null)
                    {
                        var lambda = Expression.Lambda<Func<CloudImageReport, Boolean>>(query, parameterReport);
                        if (whereExpressionReport == null)
                        {
                            whereExpressionReport = lambda;
                        }
                        else
                        {
                            whereExpressionReport = whereExpressionReport.And(lambda);
                        }
                    }
                }
                CloudImageReport report = instance.GetOne<CloudImageReport>(whereExpressionReport);

                #endregion

                #region 当没有云影像报告的时候自动生成云影像

                if (report == null)
                {
                    if (studyType != null)
                    {
                        Transaction tran = instance.GetTransaction();
                        DateTime birthday = getBirthday(info.PatientBirthday, info.PatientAge);

                        #region 患者信息（当没有患者信息时自动添加）

                        List<string> scpPatientKeys = new List<string>(archiveConfig.Keys);
                        System.Linq.Expressions.Expression<Func<CloudImagePatient, bool>> whereExpressionPatient = null;
                        ParameterExpression parameterPatient = Expression.Parameter(typeof(CloudImagePatient), "p");
                        foreach (var key in scpPatientKeys)
                        {
                            MemberExpression member = Expression.PropertyOrField(parameterPatient, key);
                            ConstantExpression constant = Expression.Constant(Convert.ChangeType(info.GetType().GetProperty(key).GetValue(info, null), member.Type));
                            BinaryExpression query = Expression.Equal(member, constant);
                            if (query != null)
                            {
                                var lambda = Expression.Lambda<Func<CloudImagePatient, Boolean>>(query, parameterPatient);
                                if (whereExpressionPatient == null)
                                {
                                    whereExpressionPatient = lambda;
                                }
                                else
                                {
                                    whereExpressionPatient = whereExpressionPatient.And(lambda);
                                }
                            }
                        }
                        CloudImagePatient patient = instance.GetOne<CloudImagePatient>(whereExpressionPatient);
                        if (patient == null)
                        {
                            patient = new CloudImagePatient()
                            {
                                ID = instance.GetID<CloudImagePatient>(),
                                PatientID = info.PatientID,
                                Name = info.PatientName,
                                Birthday = birthday,
                                Sex = (int)ElincloudServer.Common.Utils.StringUtil.ConvertDicomSexToEnumSex(info.PatientSex),
                                CreateDate = DateTime.Now,
                                UpdateDate = DateTime.Now,
                                HospitalNumber = archiveConfig.Hospital.Code
                            };
                            tran.Insert<CloudImagePatient>(patient);
                        }

                        #endregion

                        #region 报告信息

                        DateTime studyDate;
                        if (!DateTime.TryParse(info.StudyDate + " " + info.StudyTime, out studyDate))
                        {
                            studyDate = DateTime.Now;
                        }

                        report = new CloudImageReport()
                        {
                            ID = instance.GetID<CloudImageReport>(7, ((int)studyType.SysType).ToString()),
                            CloudImagePatientID = patient.ID,
                            SecurityCode = ElincloudServer.Common.Utils.StringUtil.Encrypt(ElincloudServer.Common.Utils.StringUtil.GetRandomCode(6)),
                            PatientID = info.PatientID,
                            ExaminationID = info.ExaminationID,
                            HospitalNumber = info.HospitalNumber,
                            PatientName = info.PatientName,
                            PatientAge = ElincloudServer.Common.Utils.StringUtil.ConvertDicomAgeToChineseAge(info.PatientAge),
                            PatientSex = ElincloudServer.Common.Utils.StringUtil.ConvertDicomSexToEnumSex(info.PatientSex).ToString(),
                            StudyTypeName = info.StudyTypeName,
                            StudyDate = studyDate,
                            Status = (int)ExaminationStatus.已检查,
                            CreateDate = DateTime.Now,
                            UpdateDate = DateTime.Now,
                            ReportEnabled = true,
                            SysType = (int)studyType.SysType,
                            ReportUpdateDate = DateTime.Now,
                        };
                        tran.Insert<CloudImageReport>(report);

                        #endregion

                        if (!tran.Commit())
                        {
                            throw new ServiceException("自动生成云影像失败！");
                        }
                    }
                    else
                    {
                        throw new ServiceException("自动生成云影像失败！未找到[" + info.StudyTypeName + "]对应的系统类型");
                    }
                }

                #endregion

            }
            else
            {
                throw new ServiceException("自动生成云影像的图像信息不能为空！");
            }
        }

        public void AutoGenerateStudy(DCMImageInfo dcmImageInfo, ArchiveConfigInfo archiveConfig, CheckInfo studyType)
        {
            if (dcmImageInfo != null)
            {
                SqlAccess instance = DBServices.GetHospitalDBInstance(archiveConfig.Hospital.HospitalGroupID);
                Transaction tran = instance.GetTransaction();
                List<string> scpKeys = new List<string>(archiveConfig.Keys);
                scpKeys.Add(AutoGenerateStudyRecord._ExaminationID);//防止第三方的影像号与我们库中的影像号重复，所以加上检查号一起进行判断
                AutoGenerateStudyRecord cache = new AutoGenerateStudyRecord();
                Patient patient = null;
                RISExamination e = null;

                #region 由于影像表的检查时间列是用检查时间和检查日期进行拼接存储的，用户配置的SCP中包含检查时间时需要处理下

                DCMImageInfo info = new DCMImageInfo()
                {
                    ExaminationID = dcmImageInfo.ExaminationID,
                    StudyDate = dcmImageInfo.StudyDate,
                    FileName = dcmImageInfo.FileName,
                    HospitalNumber = dcmImageInfo.HospitalNumber,
                    ImageRelaPath = dcmImageInfo.ImageRelaPath,
                    InstanceNumber = dcmImageInfo.InstanceNumber,
                    InstanceUID = dcmImageInfo.InstanceUID,
                    PatientAge = dcmImageInfo.PatientAge,
                    PatientBirthday = dcmImageInfo.PatientBirthday,
                    PatientID = dcmImageInfo.PatientID,
                    PatientName = dcmImageInfo.PatientName,
                    PatientSex = dcmImageInfo.PatientSex,
                    ReportID = dcmImageInfo.ReportID,
                    SeriesInstanceUID = dcmImageInfo.SeriesInstanceUID,
                    SeriesNumber = dcmImageInfo.SeriesNumber,
                    StudyInstanceUID = dcmImageInfo.StudyInstanceUID,
                    StudyTime = dcmImageInfo.StudyTime,
                    StudyTypeName = dcmImageInfo.StudyTypeName,
                    ScanThickness = dcmImageInfo.ScanThickness,
                    BodyPartExamined = dcmImageInfo.BodyPartExamined,
                    SeriesDescription = dcmImageInfo.SeriesDescription,
                    MachineKey = dcmImageInfo.MachineKey,
                };
                DateTime studyDate;
                if (!DateTime.TryParse(info.StudyDate + " " + info.StudyTime, out studyDate))
                {
                    studyDate = DateTime.Now;
                }
                info.StudyDate = studyDate.ToString("yyyy-MM-dd HH:mm:ss");

                #endregion

                #region 1.读取缓存信息，如果存在缓存则使用缓存，不存在则重新初始化缓存

                System.Linq.Expressions.Expression<Func<AutoGenerateStudyRecord, bool>> whereExpressionRepord = null;
                ParameterExpression parameterRepord = Expression.Parameter(typeof(AutoGenerateStudyRecord), "p");
                foreach (var key in scpKeys)
                {
                    MemberExpression member = Expression.PropertyOrField(parameterRepord, key);
                    ConstantExpression constant = Expression.Constant(Convert.ChangeType(info.GetType().GetProperty(key).GetValue(info, null), member.Type));
                    BinaryExpression query = Expression.Equal(member, constant);
                    if (query != null)
                    {
                        var lambda = Expression.Lambda<Func<AutoGenerateStudyRecord, Boolean>>(query, parameterRepord);
                        if (whereExpressionRepord == null)
                        {
                            whereExpressionRepord = lambda;
                        }
                        else
                        {
                            whereExpressionRepord = whereExpressionRepord.And(lambda);
                        }
                    }
                }
                AutoGenerateStudyRecord existCache = instance.GetOne<AutoGenerateStudyRecord>(whereExpressionRepord);

                if (existCache != null)
                {
                    cache = existCache;
                    patient = instance.GetOne<Patient>(p => p.ID == existCache.NewPatientID);
                    e = instance.GetOne<RISExamination>(p => p.ID == existCache.NewExaminationID);
                }
                else
                {
                    Type type = typeof(DCMImageInfo);
                    List<PropertyInfo> properties = type.GetProperties().ToList();
                    foreach (var key in scpKeys)
                    {
                        PropertyInfo prop = properties.FirstOrDefault(p => p.Name == key);
                        if (prop != null)
                        {
                            object v = prop.GetValue(info, null);
                            PropertyInfo property = cache.GetType().GetProperty(key);
                            if (property != null)
                            {
                                property.SetValue(cache, v.ToString(), null);
                            }
                        }
                    }
                }

                #endregion

                DoctorOrganizationRela dor = instance.GetOne<DoctorOrganizationRela>(p => p.DoctorID == archiveConfig.StudyCreateDoctor.ID);
                Organization org = instance.GetOne<Organization>(i => i.ID == dor.OrganizationID);

                #region 2.添加、更新患者信息

                DateTime birthday = getBirthday(info.PatientBirthday, info.PatientAge);
                Sexs sex = ElincloudServer.Common.Utils.StringUtil.ConvertDicomSexToEnumSex(info.PatientSex);

                if (patient == null)
                {
                    patient = new Patient()
                    {
                        ID = instance.GetID<Patient>(),
                        Name = info.PatientName,
                        Birthday = birthday,
                        Sex = (int)sex,
                        UpdateDate = DateTime.Now,
                        UpdateUserID = archiveConfig.StudyCreateDoctor.ID,
                        CreateDate = DateTime.Now,
                        CreateUserID = archiveConfig.StudyCreateDoctor.ID,
                    };
                    tran.Insert<Patient>(patient);
                }
                else
                {
                    patient.Name = info.PatientName;
                    patient.Birthday = birthday;
                    patient.Sex = (int)sex;
                    patient.UpdateDate = DateTime.Now;
                    patient.UpdateUserID = archiveConfig.StudyCreateDoctor.ID;
                    tran.Update<Patient>(patient, p => p.ID == patient.ID,
                     Patient._Name,
                     Patient._Sex,
                     Patient._UpdateDate,
                     Patient._UpdateUserID,
                     Patient._Birthday);
                }
                cache.NewPatientID = patient.ID;

                #endregion

                #region 3.添加、更新检查信息


                string patientAge = ElincloudServer.Common.Utils.StringUtil.ConvertDicomAgeToChineseAge(info.PatientAge);
                if (string.IsNullOrWhiteSpace(patientAge))
                {
                    patientAge = ElincloudServer.Common.Utils.StringUtil.ConvertChineseAgeToBirthday(birthday);
                }

                if (e == null)
                {
                    if (studyType != null)
                    {
                        string examID = instance.GetID<RISExamination>(7, ((int)SysTypes.RIS).ToString());
                        e = new RISExamination()
                        {
                            //org表由于设计因素，机构的上级必然是对应的医院节点，所以这里可以用org.pid来获取他的上级医院ID
                            ID = examID,
                            PatientID = patient.ID,
                            ApplyDoctorName = archiveConfig.StudyCreateDoctor.UserName,
                            ApplyOrgName = org.Name,
                            PatientAge = patientAge,
                            RegistDate = DateTime.Now,
                            PatientName = patient.Name,
                            PatientSex = patient.Sex,
                            StudyTypeName = info.StudyTypeName,
                            RegistDoctorName = archiveConfig.StudyCreateDoctor.UserName,
                            Status = (int)ExaminationStatus.已登记,
                            UpdateDate = DateTime.Now,
                            UpdateUserID = archiveConfig.StudyCreateDoctor.ID,
                            CreateDate = DateTime.Now,
                            CreateUserID = archiveConfig.StudyCreateDoctor.ID,
                            StudyDate = studyDate,
                            ImageExaminationID = dcmImageInfo.ExaminationID,
                            ImagePatientID = dcmImageInfo.PatientID,
                            MachineKey = dcmImageInfo.MachineKey
                        };
                        tran.Insert<RISExamination>(e);
                    }
                    else
                    {
                        throw new ServiceException("自动生成检查失败！未找到[" + info.StudyTypeName + "]对应的系统类型");
                    }
                }
                else
                {
                    e.PatientID = patient.ID;
                    e.ApplyDoctorName = archiveConfig.StudyCreateDoctor.UserName;
                    e.ApplyOrgName = org.Name;
                    e.StudyDate = studyDate;
                    e.PatientAge = patientAge;
                    e.UpdateDate = DateTime.Now;
                    e.UpdateUserID = archiveConfig.StudyCreateDoctor.ID;
                    e.ImageExaminationID = dcmImageInfo.ExaminationID;
                    e.ImagePatientID = dcmImageInfo.PatientID;
                    e.MachineKey = dcmImageInfo.MachineKey;
                    e.Status = (int)ExaminationStatus.已登记;
                    if (e.SliceImageStatus == (int)SliceImageStatus.已切割)
                    {
                        e.SliceImageStatus = (int)SliceImageStatus.正在切割;
                    }
                    else
                    {
                        e.SliceImageStatus = (int)SliceImageStatus.未切割;
                    }
                    tran.Update<RISExamination>(e, p => p.ID == existCache.NewExaminationID,
                        RISExamination._PatientID,
                        RISExamination._ApplyDoctorName,
                        RISExamination._ApplyOrgName,
                        RISExamination._StudyDate,
                        RISExamination._PatientAge,
                        RISExamination._UpdateDate,
                        RISExamination._UpdateUserID,
                        RISExamination._ImageExaminationID,
                        RISExamination._ImagePatientID,
                        RISExamination._Status,
                        RISExamination._SliceImageStatus,
                        RISExamination._MachineKey);
                }

                cache.NewExaminationID = e.ID;

                #endregion

                #region 4.更新自动登录缓存

                if (cache.ID != string.Empty)
                {
                    tran.Update<AutoGenerateStudyRecord>(cache, p => p.ID == cache.ID);
                }
                else
                {
                    cache.ID = instance.GetID<AutoGenerateStudyRecord>(7);
                    tran.Insert<AutoGenerateStudyRecord>(cache);
                }

                #endregion

                if (!tran.Commit())
                {
                    throw new ServiceException("自动登记失败！");
                }
            }
            else
            {
                throw new ServiceException("自动登记的图像信息不能为空！");
            }
        }

        /// <summary>
        /// 通过Dicom文件中生日字符串和检查年龄换算生日
        /// </summary>
        /// <param name="patientBirthday">生日（优先使用该字段换算）</param>
        /// <param name="patientAge">检查年龄（生日换算失败则用该字段换算）</param>
        /// <returns></returns>
        private DateTime getBirthday(string patientBirthday, string patientAge)
        {
            DateTime birthday = Convert.ToDateTime("1900-1-1");
            bool success = DateTime.TryParse(patientBirthday, out birthday);
            if (success && birthday.Year != 1900)
            {
                return birthday;
            }
            else
            {
                return ElincloudServer.Common.Utils.StringUtil.ConvertDicomAgeToBirthday(patientAge, null);
            }
        }

        public void AddArchiveDetails(string hospitalGroupID, List<DCMImageInfo> infos, bool IsEnableArchiveCache)
        {
            try
            {
                if (infos != null && infos.Count > 0)
                {
                    SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
                    List<ExaminationImageDetail> details = new List<ExaminationImageDetail>();
                    foreach (var info in infos)
                    {
                        DateTime studyDate;
                        if (!DateTime.TryParse(info.StudyDate + " " + info.StudyTime, out studyDate))
                        {
                            studyDate = DateTime.Now;
                        }
                        //有些图像中存储的时间格式不标准，导致存储到库中时会携带AM或者PM字样，所以使用下面代码进行格式化
                        studyDate = new DateTime(studyDate.Year, studyDate.Month, studyDate.Day, studyDate.Hour, studyDate.Minute, studyDate.Second);
                        DBEntity.ExaminationImageDetail detail = new DBEntity.ExaminationImageDetail()
                        {
                            AccessionNumber = info.ExaminationID,
                            InstanceNumber = info.InstanceNumber,
                            InstanceUID = info.InstanceUID,
                            Modality = info.StudyTypeName,
                            PatientAge = info.PatientAge,
                            PatientBirthday = info.PatientBirthday,
                            PatientID = info.PatientID,
                            PatientName = info.PatientName,
                            PatientSex = info.PatientSex,
                            StudyInstanceUID = info.StudyInstanceUID,
                            SeriesInstanceUID = info.SeriesInstanceUID,
                            StudyDate = studyDate,
                            ImageRelaPath = info.ImageRelaPath + "\\" + info.FileName,
                            SeriesNumber = info.SeriesNumber,
                            HospitalNumber = info.HospitalNumber
                        };
                        if (IsEnableArchiveCache)
                        {
                            detailDics.Add(hospitalGroupID, detail);
                        }
                        else
                        {
                            details.Add(detail);
                        }
                    }
                    if (!IsEnableArchiveCache)
                    {
                        DataTable dt = ElincloudServer.Common.Utils.DataTableUtil.ToDataTable<ExaminationImageDetail>(details);
                        //将数据插入临时表
                        instance.InsertMuch<ExaminationImageDetail>(dt, ElincloudServer.Common.ApplicationEx.ExaminationImageDetailTempTableName);
                        //调用存储过程将数据移动到正式表中
                        instance.ExecuteProcedure(ElincloudServer.Common.ApplicationEx.MoveExaminationImageDetailTempDataProcName);
                    }
                }
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("插入归档详情信息异常！", ex);
            }

        }

        public bool AddCacheDataToDB(Func<string, ArchiveConfigInfo> getArchiveConfig, Func<string, string, CheckInfo> getStudyType)
        {
            bool success = false;
            try
            {
                foreach (var key in imageDics.Keys)
                {
                    SqlAccess instance = DBServices.GetHospitalDBInstance(key);

                    List<ExaminationImage> images = imageDics[key] as List<ExaminationImage>;
                    DataTable dt = Common.Utils.DataTableUtil.ToDataTable<ExaminationImage>(images);
                    //将数据插入临时表
                    instance.InsertMuch<ExaminationImage>(dt, ElincloudServer.Common.ApplicationEx.ExaminationImageTempTableName);
                    //调用存储过程将数据移动到正式表中
                    instance.ExecuteProcedure(ElincloudServer.Common.ApplicationEx.MoveExaminationImageTempDataProcName);

                    //处理自动登记
                    if (getArchiveConfig != null && images.Count > 0)
                    {
                        List<ExaminationImage> groupedResult = images.GroupBy(item => new { item.ExaminationID, item.PatientID, item.HospitalNumber, item.StudyTypeName }).Select(group => group.FirstOrDefault()).ToList();
                        foreach (var item in groupedResult)
                        {
                            ArchiveConfigInfo configInfo = getArchiveConfig(item.HospitalNumber);
                            if (configInfo.AutoGenerateStudy)
                            {
                                CheckInfo studyType = getStudyType(item.HospitalNumber, item.StudyTypeName);
                                DCMImageInfo info = new DCMImageInfo()
                                {
                                    PatientID = item.PatientID,
                                    ExaminationID = item.ExaminationID,
                                    HospitalNumber = item.HospitalNumber,
                                    PatientAge = item.PatientAge,
                                    PatientBirthday = item.PatientBirthday,
                                    PatientName = item.PatientName,
                                    PatientSex = item.PatientSex,
                                    SeriesInstanceUID = item.SeriesInstanceUID,
                                    SeriesNumber = item.SeriesNumber,
                                    StudyDate = item.StudyDate.ToString("yyyy-MM-dd"),
                                    StudyInstanceUID = item.StudyInstanceUID,
                                    StudyTime = item.StudyDate.ToString("HH:mm:ss"),
                                    StudyTypeName = item.StudyTypeName,
                                    ImageRelaPath = item.ImageRelaPath,
                                    ScanThickness = item.ScanThickness,
                                    BodyPartExamined = item.BodyPartExamined,
                                    SeriesDescription = item.SeriesDescription,
                                    MachineKey = item.MachineKey
                                };
                                AutoGenerateStudy(info, configInfo, studyType);
                            }
                        }
                    }
                }

                foreach (var key in detailDics.Keys)
                {
                    SqlAccess instance = DBServices.GetHospitalDBInstance(key);

                    List<ExaminationImageDetail> details = detailDics[key] as List<ExaminationImageDetail>;
                    DataTable dt = Common.Utils.DataTableUtil.ToDataTable<ExaminationImageDetail>(details);
                    //将数据插入临时表
                    instance.InsertMuch<ExaminationImageDetail>(dt, ElincloudServer.Common.ApplicationEx.ExaminationImageDetailTempTableName);
                    //调用存储过程将数据移动到正式表中
                    instance.ExecuteProcedure(ElincloudServer.Common.ApplicationEx.MoveExaminationImageDetailTempDataProcName);
                }
                success = true;
            }
            catch (Exception ex)
            {
                Common.Utils.LogUtil.RecordErrorLog("插入归档详情信息异常！", ex);
            }
            finally
            {
                //错误之后会重新进行数据库写入，所以不需要保留缓存信息
                imageDics.Clear();
                detailDics.Clear();
            }
            return success;
        }

        /// <summary>
        /// 归档完成后将自动登记的检查状态改为已检查
        /// </summary>
        public bool MarkAutoGenerateStudyChecked(string hospitalGroupID, string examinationID, string patientID)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(hospitalGroupID) || string.IsNullOrWhiteSpace(examinationID) || string.IsNullOrWhiteSpace(patientID))
                {
                    return false;
                }
                SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
                AutoGenerateStudyRecord cache = instance.GetOne<AutoGenerateStudyRecord>(p => p.ExaminationID == examinationID && p.PatientID == patientID);
                if (cache == null || string.IsNullOrWhiteSpace(cache.NewExaminationID))
                {
                    return false;
                }
                RISExamination exam = instance.GetOne<RISExamination>(p => p.ID == cache.NewExaminationID);
                if (exam == null)
                {
                    return false;
                }
                if (exam.Status != (int)ExaminationStatus.已登记)
                {
                    return true;
                }
                exam.Status = (int)ExaminationStatus.已检查;
                exam.UpdateDate = DateTime.Now;
                Transaction trans = instance.GetTransaction();
                trans.Update<RISExamination>(exam, p => p.ID == exam.ID, RISExamination._Status, RISExamination._UpdateDate);
                return SqlAccess.ExecuteTransactions(trans);
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("自动登记检查状态更新异常", ex);
                return false;
            }
        }
    }
}
