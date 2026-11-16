using Dicom;
using ElincloudServer.ArchiveService;
using ElincloudServer.Entity;
using System.Reflection;
using System.Text;
using DCMImageInfo = ElincloudAISliceServer.Entity.DCMImageInfo;

namespace ElincloudServer.Common
{
    public class ArchiveUtil
    {
        public ClientResult GetArchiveTargetFileName(DCMModel model, string sourceFileFullName, string targetArchivePath, ArchiveConfigInfo archiveConfig)
        {
            string archiveRelaPath = targetArchivePath;
            foreach (var item in archiveConfig.ArchivePathRules)
            {
                Type type = model.GetType();
                List<PropertyInfo> properties = type.GetProperties().ToList();
                PropertyInfo property = properties.FirstOrDefault(p => p.Name == item.Name);
                if (property != null)
                {
                    object value = property.GetValue(model, null);
                    if (value != null && value.ToString().Trim() != string.Empty)
                    {
                        string strvalue = value.ToString().Trim();

                        if (item.DataType == ArchivePathRulesItemTypes.DateTime)
                        {
                            DateTime studyDate;
                            bool dateConvertSuccess = DateTime.TryParse(strvalue, out studyDate);
                            if (dateConvertSuccess)
                            {
                                archiveRelaPath = ElincloudServer.Common.Utils.StringUtil.Combine(archiveRelaPath, studyDate.ToString("yyyyMMdd"));
                            }
                            else
                            {
                                string contentMsg = "[" + Utils.StringUtil.GetFileName(sourceFileFullName) + "]的[日期(" + property.Name + ")]的格式不正确!";
                                return new ClientResult() { Success = false, Message = contentMsg };
                            }
                        }
                        else
                        {
                            archiveRelaPath = ElincloudServer.Common.Utils.StringUtil.Combine(archiveRelaPath, strvalue);
                        }
                    }
                    else
                    {
                        if (item.DefaultValue != string.Empty)
                        {
                            archiveRelaPath = ElincloudServer.Common.Utils.StringUtil.Combine(archiveRelaPath, item.DefaultValue);
                        }
                        else if (item.Required)
                        {
                            string contentMsg = "[" + sourceFileFullName + "]的[值(" + property.Name + ")]不能为空!";
                            return new ClientResult() { Success = false, Message = contentMsg };
                        }
                    }
                }

            }
            FileNameMode fileNameMode = FileNameMode.OriginalFileName;
            if (archiveConfig.DicomFileReNameRuleInfo != null && archiveConfig.DicomFileReNameRuleInfo.Count > 0)
            {
                DicomFileReNameRuleInfo modalityOrderByRelaInfo = archiveConfig.DicomFileReNameRuleInfo.FirstOrDefault(p => p.Modality == model.Modality);
                if (modalityOrderByRelaInfo != null)
                {
                    fileNameMode = modalityOrderByRelaInfo.FileNameMode;
                }
            }

            string targetPath = archiveRelaPath;// ElincloudServer.Common.Utils.StringUtil.Combine(ArchiveConfig.Instance.ImageTargetPath, archiveRelaPath);
            if (!System.IO.Directory.Exists(targetPath))
            {
                Common.Utils.FileUtil.CreateDirectory(targetPath);
            }
            string fileName = Utils.StringUtil.GetFileName(sourceFileFullName);
            if (fileNameMode == FileNameMode.InstanceUID && model.SOPInstanceUID != string.Empty)
            {
                fileName = model.SOPInstanceUID + ".dcm";
            }
            else if (fileNameMode == FileNameMode.InstanceNumber && model.InstanceNumber != string.Empty)
            {
                //为了解决现场部分数据存在 同一个序列下有InstanceNumber重复的情况
                //fileName = model.InstanceNumber + ".dcm";
                fileName = model.InstanceNumber +"_"+ model.SOPInstanceUID + ".dcm";
            }
            else
            {
                fileName = Utils.StringUtil.GetFileNameWithoutExtension(sourceFileFullName) + ".dcm";
            }
            string targetFileName = ElincloudServer.Common.Utils.StringUtil.Combine(targetPath, fileName);
            return new ClientResult() { Success = true, Result = targetFileName };
        }

        public ClientResult SaveDcmFileAndGetDcmImageInfo(string sourceFillFullName, string targetFileFullName, string imageTargetPath, DCMModel model)
        {
            if (model.PatientID != string.Empty)
            {
                //如果目标地有同名文件 则覆盖文件,如果没有再移动文件到目标目录

                if (!System.IO.File.Exists(targetFileFullName))
                {
                    System.IO.File.Move(sourceFillFullName, targetFileFullName);
                }
                else
                {
                    System.IO.File.Copy(sourceFillFullName, targetFileFullName, true);
                    System.IO.File.Delete(sourceFillFullName);
                }
                string fileName = Utils.StringUtil.GetFileName(targetFileFullName);
                string archiveRelaPath = Utils.StringUtil.GetDirectoryName(targetFileFullName).Replace(imageTargetPath, "");

                #region 移除archiveRelaPath路径前面的斜杠

                // 同时匹配正斜杠和反斜杠
                char[] separators = { '/', '\\' };
                // 移除非空字符串开头处的所有分隔符
                int start = 0;
                while (start < archiveRelaPath.Length && separators.Contains(archiveRelaPath[start]))
                {
                    start++;
                }
                archiveRelaPath = start == 0 ? archiveRelaPath : archiveRelaPath.Substring(start);

                #endregion

                DCMImageInfo info = new DCMImageInfo()
                {
                    PatientID = model.PatientID,
                    ExaminationID = model.AccessionNumber == string.Empty ? model.PatientID : model.AccessionNumber,//如果检查号为空则用影像号替代20181115
                    StudyInstanceUID = model.StudyInstanceUID,
                    SeriesInstanceUID = model.SeriesInstanceUID,
                    SeriesNumber = model.SeriesNumber,
                    PatientName = model.PatientName,
                    PatientAge = model.PatientAge,
                    PatientBirthday = model.PatientBirthDate,
                    PatientSex = model.PatientSex,
                    InstanceUID = model.SOPInstanceUID,
                    InstanceNumber = model.InstanceNumber,
                    StudyTypeName = model.Modality,
                    StudyDate = model.StudyDate,
                    StudyTime = model.StudyTime,
                    ImageRelaPath = archiveRelaPath,
                    FileName = fileName,
                    HospitalNumber = model.HospitalNumber,
                    MachineKey = model.MachineKey,
                    ScanThickness = model.ScanThickness,
                    BodyPartExamined = model.BodyPartExamined,
                    SeriesDescription = model.SeriesDescription,
                };
                return new ClientResult() { Success = true, Result = info };
            }
            else
            {
                string contentMsg = model.PatientID == string.Empty ? "影像号" : "";
                return new ClientResult() { Success = false, Message = contentMsg };
            }
        }

        public DCMModel GetModel(string dcmFileFullName)
        {
            DicomFile df = DicomFile.Open(dcmFileFullName, FileReadOption.SkipLargeTags);
            if (df != null)
            {
                string modality = df.Dataset.GetSingleValueOrDefault<string>(Dicom.DicomTag.Modality, "");//检查类型
                                                                                                          //PR图像后期在影像工作站有对应的功能，所以此处不进行过滤
                if (modality.ToUpper() != "PR")
                {
                    int bitsAllocated = df.Dataset.GetSingleValueOrDefault<int>(DicomTag.BitsAllocated, 0);
                    if (bitsAllocated == 0)
                    {
                        return null;
                    }
                }

                return GetModel(df.Dataset);
            }
            else
            {
                return null;
            }
        }

        public DCMModel GetModel(DicomDataset dataset)
        {
            try
            {
                if (dataset != null)
                {
                    DCMModel model = new DCMModel();
                    Type type = model.GetType();
                    PropertyInfo[] properties = type.GetProperties();
                    foreach (var p in properties)
                    {
                        Dicom.DicomTag tag = DCMModel.DcmTags[p.Name];
                        string value = "";
                        if (DicomTag.StudyDate.Group == tag.Group && DicomTag.StudyDate.Element == tag.Element)
                        {
                            DateTime dt = dataset.GetSingleValueOrDefault<DateTime>(tag, DateTime.Now.Date);
                            value = dt.ToString("yyyy-MM-dd");
                        }
                        else if (DicomTag.StudyTime.Group == tag.Group && DicomTag.StudyTime.Element == tag.Element)
                        {
                            DateTime dt = dataset.GetSingleValueOrDefault<DateTime>(tag, DateTime.Now.Date);
                            value = dt.ToString("HH:mm:ss");
                        }
                        else if (DicomTag.PatientBirthDate.Group == tag.Group && DicomTag.PatientBirthDate.Element == tag.Element)
                        {
                            DateTime dt = dataset.GetSingleValueOrDefault<DateTime>(tag, new DateTime(1900, 1, 1));
                            DateTime dttime = dataset.GetSingleValueOrDefault<DateTime>(DicomTag.PatientBirthTime, DateTime.Now.Date);
                            value = dt.ToString("yyyy-MM-dd") + " " + dttime.ToString("HH:mm:ss");

                            if (dt.Year == 1900)
                            {
                                DateTime StudyDate = dataset.GetSingleValueOrDefault<DateTime>(DicomTag.StudyDate, DateTime.Now.Date);
                                string PatientAge = dataset.GetSingleValueOrDefault<string>(DicomTag.PatientAge, "");
                                value = ElincloudServer.Common.Utils.StringUtil.ConvertDicomAgeToBirthday(PatientAge, StudyDate).ToString("yyyy-MM-dd HH:mm:ss");
                            }
                        }
                        else if (DicomTag.PatientName.Group == tag.Group && DicomTag.PatientName.Element == tag.Element)
                        {
                            //患者姓名可能出现中文 需要进行编码格式转换 目前已经测试通过ascii 和 ISO-8859-1
                            //TODO:之前王友那里有两张患者名字是乱码的图 采用else中的逻辑可以正常转码，老季发的UTF8的图不行，现在要把这三张图拿到一起综合测试一下。输出一个工具类来判断字节数组是什么编码格式 然后用指定的字符集来转码

                            byte[] patientNameBytes = dataset.GetValues<byte>(DicomTag.PatientName);
                            if (ElincloudServer.Common.Utils.StringUtil.IsTextUTF8(patientNameBytes))
                            {
                                value = Encoding.UTF8.GetString(patientNameBytes);
                            }
                            else
                            {
                                if (ElincloudServer.Common.Utils.StringUtil.IsTextGB18030(patientNameBytes))
                                {
                                    value = Encoding.GetEncoding("GB18030").GetString(patientNameBytes);
                                }
                                else
                                {
                                    //value = Encoding.Default.GetString(patientNameBytes);
                                    value = dataset.GetSingleValue<string>(DicomTag.PatientName);
                                }
                            }
                        }
                        else if (DicomTag.StudyInstanceUID.Group == tag.Group && DicomTag.StudyInstanceUID.Element == tag.Element)
                        {
                            string StudyInstanceUID = string.Empty;
                            dataset.TryGetString(tag, out StudyInstanceUID);
                            value = ElincloudServer.Common.Utils.StringUtil.GetDicomUID(StudyInstanceUID);
                        }
                        else if (DicomTag.SeriesInstanceUID.Group == tag.Group && DicomTag.SeriesInstanceUID.Element == tag.Element)
                        {
                            string SeriesInstanceUID = string.Empty;
                            dataset.TryGetString(tag, out SeriesInstanceUID);
                            value = ElincloudServer.Common.Utils.StringUtil.GetDicomUID(SeriesInstanceUID);
                        }
                        else if (DicomTag.Modality.Group == tag.Group && DicomTag.Modality.Element == tag.Element)
                        {
                            string Modality = string.Empty;
                            dataset.TryGetString(tag, out Modality);
                            value = ElincloudServer.Common.Utils.StringUtil.GetDicomModality(Modality);
                        }
                        else
                        {
                            value = dataset.GetSingleValueOrDefault<string>(tag, "");
                        }
                        if (!string.IsNullOrWhiteSpace(value) && value.Contains("\0"))
                        {
                            value = value.Replace("\0", "");
                        }
                        p.SetValue(model, value == null ? "" : value.Trim(), null);
                    }
                    return model;
                }
                else
                {
                    return null;
                }
            }
            catch (AccessViolationException ex)
            {
                ArchiveConfigUtil.Instance.PrintLogAction("读取DCM图像信息异常", true, null, ex);
                return null;
            }
            catch (Exception ex)
            {
                ArchiveConfigUtil.Instance.PrintLogAction("读取DCM图像信息异常", true, null, ex);
                return null;
            }
        }
    }
}
