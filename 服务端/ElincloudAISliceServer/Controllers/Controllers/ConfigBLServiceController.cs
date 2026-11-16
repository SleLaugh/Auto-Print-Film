using ElincloudServer.BLServices;
using ElincloudServer.Common;
using ElincloudServer.Entity;
using Microsoft.AspNetCore.Mvc;
using WorkFilmConfigInfo = ElincloudAISliceServer.Entity.WorkFilmConfigInfo;
using WorkFilmSettingsConfigInfo = ElincloudAISliceServer.Entity.WorkFilmSettingsConfigInfo;
using DicomMachineConfigInfo = ElincloudAISliceServer.Entity.DicomMachineConfigInfo;

namespace ElincloudServer.Controllers
{
    public class ConfigBLServiceController : BaseController
    {
        #region 启动服务配置

        public JsonResult GetServerConfigInfo()
        {
            return Excute(new Func<List<ServiceInfo>>(() =>
            {
                List<ServiceInfo> rets = new List<ServiceInfo>();
                ServerConfigInfo configinfo = BLService.ConfigBLService.GetServerConfigInfo();
                if (configinfo != null && configinfo.ServiceNames != null)
                {
                    foreach (var item in configinfo.ServiceNames)
                    {
                        item.ServiceStatus = Common.Utils.ProcessUtil.CheckProcess(item.ServiceENName) ? ForwardingSystemServiceStatus.运行中 : ForwardingSystemServiceStatus.未运行;
                    }
                    rets = configinfo.ServiceNames;
                }
                return rets;
            }), "E00000105");
        }


        public JsonResult SetServerConfigInfo(ServerConfigInfo configInfo)
        {
            return Excute(new Action(() =>
            {
                BLService.ConfigBLService.SetServerConfigInfo(configInfo);
            }), "E00000106");
        }
        #endregion

        #region 关于我们
        /// <summary>
        /// 获取关于我们信息
        /// </summary>
        /// <param name="hospitalID">医院ID</param>
        /// <returns>{"Result":"AboutConfigInfo","Success":true,"Message":""}</returns>
        public JsonResult GetAboutConfigInfo(string hospitalID)
        {
            return Excute<AboutConfigInfo>(new Func<PermissionInfo, AboutConfigInfo>((PermissionInfo pi) =>
            {
                return BLService.ConfigBLService.GetAboutConfigInfo(hospitalID);
            }), "E000003");
        }
        /// <summary>
        /// 设置关于我们信息
        /// </summary>
        /// <param name="aboutConfigInfo">AboutConfigInfo实体</param>
        /// <returns>{"Result":null,"Success":true,"Message":""}</returns>
        public JsonResult SetAboutConfigInfo(AboutConfigInfo aboutConfigInfo)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.ConfigBLService.SetAboutConfigInfo(aboutConfigInfo);
            }), "E000004");
        }

        #endregion

        #region 医生客户端配置

        public JsonResult GetDoctorClientConfigInfos(string hospitalID, List<string> uids)
        {
            return Excute<List<DoctorClientConfigInfo>>(new Func<PermissionInfo, List<DoctorClientConfigInfo>>((PermissionInfo pi) =>
            {
                return BLService.ConfigBLService.GetDoctorClientConfigInfos(hospitalID, uids);
            }), "E000005");
        }

        public JsonResult SetDoctorClientConfigInfo(DoctorClientConfigInfo config)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.ConfigBLService.SetDoctorClientConfigInfo(config);
            }), "E000006");
        }

        public JsonResult SetDoctorClientConfigInfos(List<DoctorClientConfigInfo> configs)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.ConfigBLService.SetDoctorClientConfigInfos(configs);
            }), "E000006");
        }

        #endregion

        #region 服务器公共配置

        public JsonResult GetServiceCommonConfigInfo()
        {
            return Excute(new Func<ServiceCommonConfigInfo>(() =>
            {
                return BLService.ConfigBLService.GetServiceCommonConfigInfo();
            }), "E0000012");
        }

        public JsonResult SetServiceCommonConfigInfo(ServiceCommonConfigInfo configInfo)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.ConfigBLService.SetServiceCommonConfigInfo(configInfo);
            }), "E0000013");
        }

        public JsonResult GetHospitalTreeNodeInfo()
        {
            return Excute<List<HospitalTreeNodeInfo>>(new Func<PermissionInfo, List<HospitalTreeNodeInfo>>((PermissionInfo pi) =>
            {
                return BLService.ConfigBLService.GetHospitalTreeNodeInfo(pi.HospitalIDs);
            }), "E00000581");
        }
        #endregion

        #region CStore配置

        public JsonResult GetCStoreConfigInfoByHospitalCode(string hospitalCode)
        {
            return Excute(new Func<PermissionInfo, CStoreConfigInfo>((PermissionInfo pi) =>
            {
                return BLService.ConfigBLService.GetCStoreConfigInfoByHospitalCode(hospitalCode);
            }), "E0000010");
        }

        public JsonResult SetCStoreConfigInfo(CStoreConfigInfo configInfo)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.ConfigBLService.SetCStoreConfigInfo(configInfo);
            }), "E0000011");
        }

        #endregion

        #region Dicom设备通信配置

        public JsonResult GetDicomMachineInfoByHospitalCode(string hospitalCode)
        {
            return Excute<DicomMachineConfigInfo>(new Func<PermissionInfo, DicomMachineConfigInfo>((PermissionInfo pi) =>
            {
                return BLService.ConfigBLService.GetDicomMachineInfoByHospitalCode(hospitalCode);
            }), "E0000017");
        }

        public JsonResult SetDicomMachineInfo(DicomMachineConfigInfo configInfo)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.ConfigBLService.SetDicomMachineInfo(configInfo);
            }), "E0000018");
        }

        #endregion

        #region 归档配置
        public JsonResult GetArchiveConfigInfos()
        {
            return Excute<List<ArchiveConfigInfo>>(new Func<PermissionInfo, List<ArchiveConfigInfo>>((PermissionInfo pi) =>
            {
                return BLService.ConfigBLService.GetArchiveConfigInfos();
            }), "E000007");
        }

        public JsonResult GetArchiveConfigInfoByHospitalCode(string hospitalCode)
        {
            return Excute(new Func<PermissionInfo, ArchiveConfigInfo>((PermissionInfo pi) =>
            {
                return BLService.ConfigBLService.GetArchiveConfigInfoByHospitalCode(hospitalCode);
            }), "E000008");
        }

        public JsonResult SetArchiveConfigInfo(List<ArchiveConfigInfo> configInfos)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.ConfigBLService.SetArchiveConfigInfo(configInfos);
            }), "E000009");
        }

        #endregion

        #region 数据存储服务配置
        public JsonResult GetDataStoreConfigInfo()
        {
            return Excute<DataStoreConfigInfo>(new Func<PermissionInfo, DataStoreConfigInfo>((PermissionInfo pi) =>
            {
                return BLService.ConfigBLService.GetDataStoreConfigInfo();
            }), "E0000051");
        }

        public JsonResult SetDataStoreConfigInfo(DataStoreConfigInfo configInfo)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.ConfigBLService.SetDataStoreConfigInfo(configInfo);
            }), "E0000052");
        }
        #endregion

        #region 清除文件配置
        /// <summary>
        /// 返回缓存子文件夹目录
        /// </summary>
        /// <returns></returns>
        public JsonResult GetCacheSubfolders()
        {
            return Excute<List<KeyValueModel>>(new Func<PermissionInfo, List<KeyValueModel>>((PermissionInfo pi) =>
            {
                return BLService.ConfigBLService.GetCacheSubfolders();
            }), "E0000053");
        }

        public JsonResult GetClearFileConfigInfo()
        {
            return Excute<ClearFileConfigInfo>(new Func<PermissionInfo, ClearFileConfigInfo>((PermissionInfo pi) =>
            {
                return BLService.ConfigBLService.GetClearFileConfigInfo();
            }), "E0000053");
        }

        public JsonResult SetClearConfigInfo(ClearFileConfigInfo configInfo)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.ConfigBLService.SetClearConfigInfo(configInfo);
            }), "E0000054");
        }
        #endregion


        #region 数据库备份服务
        public JsonResult GetDBBackupConfigInfo()
        {
            return Excute<DBBackupConfigInfo>(new Func<PermissionInfo, DBBackupConfigInfo>((PermissionInfo pi) =>
            {
                return BLService.ConfigBLService.GetDBBackupConfigInfo();
            }), "E0000061");
        }

        public JsonResult SetDBBackupConfigInfo(DBBackupConfigInfo configInfo)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.ConfigBLService.SetDBBackupConfigInfo(configInfo);
            }), "E0000062");
        }

        public JsonResult GetDBNames()
        {
            return Excute<List<string>>(new Func<PermissionInfo, List<string>>((PermissionInfo pi) =>
            {
                return BLService.ConfigBLService.GetDBNames();
            }), "E0000063");
        }
        #endregion

        #region 影像工作站配置
        public string LoginName
        {
            get
            {
                string userName = Service.GetHeaderValue(ApplicationEx.UserName);
                return string.IsNullOrEmpty(userName) ? "default" : userName;
            }
        }

        #region 配置信息
        public JsonResult GetWorkFilmConfigInfo()
        {
            return Excute(new Func<WorkFilmConfigInfo>(() =>
            {
                WorkFilmFourCornerConfigInfo fourCorner = BLService.ConfigBLService.GetWorkFilmFourCornerConfigInfoByLoginName(LoginName);
                WorkFilmSettingsConfigInfo settings = BLService.ConfigBLService.GetWorkFilmSettingsConfigInfoByLoginName(LoginName);
                return new WorkFilmConfigInfo()
                {
                    FourCornerTemplate = fourCorner?.FourCorners,
                    GeneralSettingsInfo = settings?.Settings
                };

            }), "E0000095");
        }
        #endregion

        #region 四角信息
        public JsonResult GetWorkFilmFourCornerConfigInfo()
        {
            return Excute(new Func<WorkFilmFourCornerConfigInfo>(() =>
            {
                return BLService.ConfigBLService.GetWorkFilmFourCornerConfigInfoByLoginName(LoginName);
            }), "E0000086");
        }

        public JsonResult SetWorkFilmFourCornerConfigInfo(WorkFilmFourCornerConfigInfo configInfo)
        {
            return Excute(new Action(() =>
            {
                configInfo.LoginName = LoginName;
                BLService.ConfigBLService.SetWorkFilmFourCornerConfigInfo(configInfo);
            }), "E0000092");
        }
        #endregion

        #region 通用设置
        public JsonResult GetWorkFilmSettingsConfigInfo()
        {
            return Excute(new Func<WorkFilmSettingsConfigInfo>(() =>
            {
                return BLService.ConfigBLService.GetWorkFilmSettingsConfigInfoByLoginName(LoginName);
            }), "E0000086");
        }

        public JsonResult SetWorkFilmSettingsConfigInfo(WorkFilmSettingsConfigInfo configInfo)
        {
            return Excute(new Action(() =>
            {
                configInfo.LoginName = LoginName;
                BLService.ConfigBLService.SetWorkFilmSettingsConfigInfo(configInfo);
            }), "E0000092");
        }
        #endregion
        #endregion

        #region 密码强度验证配置
        /// <summary>
        /// 获取密码配置管理信息
        /// </summary>
        /// <param name="hospitalCode">医院code</param>
        /// <returns>{"Result":"PasswordPolicyConfigInfo","Success":true,"Message":""}</returns>
        public JsonResult GetPasswordPolicyConfigInfoByHospitalCode(string hospitalCode)
        {
            return Excute<PasswordPolicyConfigInfo>(new Func<PermissionInfo, PasswordPolicyConfigInfo>((PermissionInfo pi) =>
            {
                return BLService.ConfigBLService.GetPasswordPolicyConfigInfoByHospitalCode(hospitalCode);
            }), "E00000554");
        }
        /// <summary>
        /// 设置密码配置管理信息
        /// </summary>
        /// <param name="configInfo">PasswordPolicyConfigInfo实体</param>
        /// <returns>{"Result":null,"Success":true,"Message":""}</returns>
        public JsonResult SetPasswordPolicyConfigInfo(PasswordPolicyConfigInfo configInfo)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.ConfigBLService.SetPasswordPolicyConfigInfo(configInfo);
            }), "E00000555");
        }
        #endregion

        #region 返回文件访问地址
        public JsonResult GetFileSeriviceUrl(string accessIP)
        {
            return Excute(new Func<string>(() =>
            {
                return BLService.ConfigBLService.GetFileSeriviceUrl(accessIP);
            }), "E0000059");
        }
        #endregion

        #region 返回影像工作站访问地址
        public JsonResult GetDicomViewerSeriviceUrl(string accessIP)
        {
            return Excute(new Func<string>(() =>
            {
                return BLService.ConfigBLService.GetDicomViewerSeriviceUrl(accessIP);
            }), "E0000059");
        }
        #endregion
    }
}

