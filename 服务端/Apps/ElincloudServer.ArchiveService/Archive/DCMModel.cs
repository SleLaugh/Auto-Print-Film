
namespace ElincloudServer.Entity
{
    public class DCMModel
    {
        public static Dictionary<string, Dicom.DicomTag> DcmTags = new Dictionary<string, Dicom.DicomTag>()
        {
            {"PatientID",Dicom.DicomTag.PatientID},
            {"AccessionNumber",Dicom.DicomTag.AccessionNumber},
            {"StudyInstanceUID",Dicom.DicomTag.StudyInstanceUID},
            {"SeriesInstanceUID",Dicom.DicomTag.SeriesInstanceUID},
            {"SeriesNumber",Dicom.DicomTag.SeriesNumber},
            {"PatientName",Dicom.DicomTag.PatientName},
            {"PatientBirthDate",Dicom.DicomTag.PatientBirthDate},
            {"PatientAge",Dicom.DicomTag.PatientAge},
            {"PatientSex",Dicom.DicomTag.PatientSex},
            {"Modality",Dicom.DicomTag.Modality},
            {"SOPInstanceUID",Dicom.DicomTag.SOPInstanceUID},
            {"InstanceNumber",Dicom.DicomTag.InstanceNumber},
            {"StudyDate",Dicom.DicomTag.StudyDate},
            {"StudyTime",Dicom.DicomTag.StudyTime},
            {"NumberOfFrames",Dicom.DicomTag.NumberOfFrames},
            {"HospitalNumber",new Dicom.DicomTag(0x7211,0x0010)},
            {"MachineKey",new Dicom.DicomTag(0x7211,0x0011)},
            {"ScanThickness",Dicom.DicomTag.SliceThickness},
            {"BodyPartExamined",Dicom.DicomTag.BodyPartExamined},
            {"SeriesDescription",Dicom.DicomTag.SeriesDescription},
        };

        public string PatientID { get; set; }
        public string AccessionNumber { get; set; }
        public string StudyInstanceUID { get; set; }
        public string SeriesInstanceUID { get; set; }
        public string SeriesNumber { get; set; }
        public string PatientName { get; set; }
        public string PatientBirthDate { get; set; }
        public string PatientAge { get; set; }
        public string PatientSex { get; set; }
        public string Modality { get; set; }
        public string SOPInstanceUID { get; set; }
        public string InstanceNumber { get; set; }
        public string StudyDate { get; set; }
        public string StudyTime { get; set; }
        public string NumberOfFrames { get; set; }
        public string HospitalNumber { get; set; }
        public string MachineKey { get; set; }
        public string ScanThickness { get; set; }
        public string BodyPartExamined { get; set; }
        public string SeriesDescription { get; set; }
        

        public DCMModel Clone()
        {
            DCMModel newModel = new DCMModel()
            {
                PatientID = this.PatientID,
                AccessionNumber = this.AccessionNumber,
                StudyInstanceUID = this.StudyInstanceUID,
                SeriesInstanceUID = this.SeriesInstanceUID,
                SeriesNumber = this.SeriesNumber,
                PatientName = this.PatientName,
                PatientBirthDate = this.PatientBirthDate,
                PatientAge = this.PatientAge,
                PatientSex = this.PatientSex,
                Modality = this.Modality,
                SOPInstanceUID = this.SOPInstanceUID,
                InstanceNumber = this.InstanceNumber,
                StudyDate = this.StudyDate,
                StudyTime = this.StudyTime,
                NumberOfFrames = this.NumberOfFrames,
                HospitalNumber = this.HospitalNumber,
                MachineKey = this.MachineKey,
                BodyPartExamined = this.BodyPartExamined,
                ScanThickness = this.ScanThickness,
                SeriesDescription = this.SeriesDescription
            };
            return newModel;
        }
    }
}
