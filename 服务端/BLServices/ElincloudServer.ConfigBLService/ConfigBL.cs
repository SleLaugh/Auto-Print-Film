using ElincloudServer.Common;
using ElincloudServer.CommonBLService;
using ElincloudServer.DBEntity;
using ElincloudServer.DBService;
using ElincloudServer.Entity;
using WorkFilmSettingsConfigInfo = ElincloudAISliceServer.Entity.WorkFilmSettingsConfigInfo;
using DicomMachineConfigInfo = ElincloudAISliceServer.Entity.DicomMachineConfigInfo;

namespace ElincloudServer.ConfigBLService
{
    public class ConfigBL
    {
        #region 启动服务配置

        public ServerConfigInfo GetServerConfigInfo()
        {
            return CommonBL.GetServerConfigInfo();
        }

        public void SetServerConfigInfo(ServerConfigInfo configInfo)
        {
            if (configInfo != null)
            {
                List<ServerConfig> infos = DBServices.ElincloudConfigInstance.GetMuch<ServerConfig>();
                if (infos != null && infos.Count > 0)
                {
                    ServerConfig config = infos.FirstOrDefault();

                    // 1. 解析旧配置（数据库中的）
                    List<ServiceInfo> oldServiceList = new List<ServiceInfo>();
                    if (!string.IsNullOrEmpty(config.ServiceNameConfig))
                    {
                        ServerConfigInfo oldConfigInfo = ElincloudServer.Common.Utils.JsonUtil.FromJson<ServerConfigInfo>(config.ServiceNameConfig);
                        oldServiceList = oldConfigInfo?.ServiceNames ?? new List<ServiceInfo>();
                    }

                    // 2. 新配置（传入的 configInfo）
                    List<ServiceInfo> newServiceList = configInfo.ServiceNames;

                    // 3. 找出需要【停用】的服务：存在于旧配置，但不在新配置中
                    var toDisable = oldServiceList
                        .Where(old => !newServiceList.Any(n => n.ServiceENName == old.ServiceENName))
                        .ToList();

                    // 4. 找出需要【启用】的服务：存在于新配置，但不在旧配置中
                    var toEnable = newServiceList
                        .Where(n => !oldServiceList.Any(old => old.ServiceENName == n.ServiceENName))
                        .ToList();

                    // 5. 执行停用
                    foreach (var service in toDisable)
                    {
                        CommonBL.SetAotuStartService(service, false);
                    }

                    // 6. 执行启用
                    foreach (var service in toEnable)
                    {
                        CommonBL.SetAotuStartService(service, true);
                    }

                    // 7. 更新数据库
                    config.ServiceNameConfig = ElincloudServer.Common.Utils.JsonUtil.ToJson(configInfo);
                    DBServices.ElincloudConfigInstance.Update<ServerConfig>(config, p => p.ID == config.ID, ServerConfig._ServiceNameConfig);
                }
                else
                {
                    // 添加服务自启
                    foreach (ServiceInfo info in configInfo.ServiceNames)
                    {
                        CommonBL.SetAotuStartService(info, true);
                    }
                    ServerConfig config = new ServerConfig()
                    {
                        ID = DBServices.ElincloudConfigInstance.GetID<ServerConfig>(),
                        ServiceNameConfig = ElincloudServer.Common.Utils.JsonUtil.ToJson(configInfo),
                    };
                    DBServices.ElincloudConfigInstance.Insert<ServerConfig>(config);
                }
                //更新缓存
                CommonBL.GetServerConfigInfo(true);
            }
            else
            {
                throw new ServiceException("W0000062");
            }
        }
        #endregion

        #region 关于我们

        public AboutConfigInfo GetAboutConfigInfo(string hospitalID)
        {
            List<AboutConfig> infos = DBServices.ElincloudConfigInstance.GetMuch<AboutConfig>();
            if (infos.Count > 0)
            {
                AboutConfig config = infos.FirstOrDefault();
                AboutConfigInfo info = new AboutConfigInfo()
                {
                    AfterSaleServiceCompanyName = config.AfterSaleServiceCompanyName,
                    CompanyAddress = config.CompanyAddress,
                    CompanyName = config.CompanyName,
                    CompanySimpleName = config.CompanySimpleName,
                    FullVersion = ElincloudServer.Common.Utils.ConfigUtil.Version,
                    IsShowAbout = config.IsShowAbout,
                    ManufactureAddress = config.ManufactureAddress,
                    ManufactureDate = config.ManufactureDate,
                    ManufactureLicence = config.ManufactureLicence,
                    ModuleName = config.ModuleName,
                    PhoneNumber = config.PhoneNumber,
                    PostCode = config.PostCode,
                    ProductName = config.ProductName,
                    RegistCode = config.RegistCode,
                    ReleaseVersion = config.ReleaseVersion,
                    RemainDays = "",
                    Remark = config.Remark,
                    Specification = config.Specification,
                    Tip = config.Tip,
                };
                HospitalRegist hospitalRegist = DBServices.ElincloudBasicInstance.GetOne<HospitalRegist>(c => c.HospitalID == hospitalID);
                if (hospitalRegist != null)
                {
                    if (hospitalRegist.RegistCode == string.Empty)
                    {
                        info.RemainDays = "未授权";
                    }
                    else
                    {
                        int remaindays = ElincloudServer.Common.ApplicationEx.GetRemainDays(hospitalRegist.RegistCode, hospitalRegist.RegistKey);
                        if (remaindays >= 0)
                        {
                            info.RemainDays = remaindays + "天";
                        }
                        else
                        {
                            info.RemainDays = "授权到期";
                        }
                    }
                }
                return info;
            }
            else
            {
                return new AboutConfigInfo();
            }
        }

        public void SetAboutConfigInfo(AboutConfigInfo configInfo)
        {
            if (configInfo != null)
            {
                List<AboutConfig> infos = DBServices.ElincloudConfigInstance.GetMuch<AboutConfig>();
                if (infos.Count > 0)
                {
                    ElincloudServer.Common.Utils.ConfigUtil.Version = configInfo.FullVersion;
                    AboutConfig config = infos.FirstOrDefault();
                    config.CompanySimpleName = configInfo.CompanySimpleName;
                    config.AfterSaleServiceCompanyName = configInfo.AfterSaleServiceCompanyName;
                    config.PhoneNumber = configInfo.PhoneNumber;
                    config.PostCode = configInfo.PostCode;
                    config.IsShowAbout = configInfo.IsShowAbout;
                    DBServices.ElincloudConfigInstance.Update<AboutConfig>(config, p => p.ID == config.ID,
                        AboutConfig._CompanySimpleName,
                        AboutConfig._AfterSaleServiceCompanyName,
                        AboutConfig._PhoneNumber,
                        AboutConfig._PostCode,
                        AboutConfig._IsShowAbout);
                }
                else
                {
                    throw new ServiceException("W000003");
                }
            }
            else
            {
                throw new ServiceException("W000004");
            }
        }

        #endregion

        #region 客户端及医生个人配置

        public List<DoctorClientConfigInfo> GetDoctorClientConfigInfos(string hospitalID, List<string> uids)
        {
            List<DoctorClientConfigInfo> infos = new List<DoctorClientConfigInfo>();

            List<DoctorClientConfig> configs = DBServices.ElincloudConfigInstance.GetMuch<DoctorClientConfig>(p => p.HospitalID == hospitalID && uids.Contains(p.UID));
            if (configs.Count > 0)
            {
                foreach (var config in configs)
                {
                    infos.Add(new DoctorClientConfigInfo()
                    {
                        HospitalID = hospitalID,
                        UID = config.UID,
                        ConfigKey = config.ConfigKey,
                        ConfigValue = config.ConfigValue,
                    });
                }
            }
            return infos;
        }

        public void SetDoctorClientConfigInfo(DoctorClientConfigInfo configInfo)
        {
            if (configInfo != null)
            {
                DoctorClientConfig cc = DBServices.ElincloudConfigInstance.GetOne<DoctorClientConfig>(p => p.HospitalID == configInfo.HospitalID && p.ConfigKey == configInfo.ConfigKey && p.UID == configInfo.UID);
                if (cc != null)
                {
                    cc.ConfigValue = configInfo.ConfigValue;
                    DBServices.ElincloudConfigInstance.Update<DoctorClientConfig>(cc, p => p.HospitalID == configInfo.HospitalID && p.ConfigKey == configInfo.ConfigKey && p.UID == configInfo.UID, DoctorClientConfig._ConfigValue);
                }
                else
                {
                    cc = new DoctorClientConfig()
                    {
                        HospitalID = configInfo.HospitalID,
                        ConfigKey = configInfo.ConfigKey,
                        ConfigValue = configInfo.ConfigValue,
                        UID = configInfo.UID,
                    };
                    DBServices.ElincloudConfigInstance.Insert<DoctorClientConfig>(cc);
                }
            }
            else
            {
                throw new ServiceException("W000005");
            }
        }

        public void SetDoctorClientConfigInfos(List<DoctorClientConfigInfo> configInfos)
        {
            if (configInfos != null && configInfos.Count > 0)
            {
                List<DoctorClientConfig> configs = DBServices.ElincloudConfigInstance.GetMuch<DoctorClientConfig>();
                Transaction trans = DBServices.ElincloudConfigInstance.GetTransaction();
                foreach (var configInfo in configInfos)
                {
                    DoctorClientConfig cc = configs.FirstOrDefault((p => p.HospitalID == configInfo.HospitalID && p.ConfigKey == configInfo.ConfigKey && p.UID == configInfo.UID));
                    if (cc != null)
                    {
                        cc.ConfigValue = configInfo.ConfigValue;
                        trans.Update<DoctorClientConfig>(cc, p => p.HospitalID == configInfo.HospitalID && p.ConfigKey == configInfo.ConfigKey && p.UID == configInfo.UID, DoctorClientConfig._ConfigValue);
                    }
                    else
                    {
                        cc = new DoctorClientConfig()
                        {
                            HospitalID = configInfo.HospitalID,
                            ConfigKey = configInfo.ConfigKey,
                            ConfigValue = configInfo.ConfigValue,
                            UID = configInfo.UID,
                        };
                        trans.Insert<DoctorClientConfig>(cc);
                    }
                }
                if (!trans.Commit())
                {
                    throw new ServiceException("E000006");
                }
            }
            else
            {
                throw new ServiceException("W000005");
            }
        }

        #endregion

        #region 服务器公共配置

        public ServiceCommonConfigInfo GetServiceCommonConfigInfo()
        {
            return CommonBL.GetServiceCommonConfigInfo();
        }

        public void SetServiceCommonConfigInfo(ServiceCommonConfigInfo configInfo)
        {
            if (configInfo != null)
            {
                List<ServiceCommonConfig> infos = DBServices.ElincloudConfigInstance.GetMuch<ServiceCommonConfig>();
                if (infos.Count > 0)
                {
                    ServiceCommonConfig config = infos.FirstOrDefault();
                    config.CommonConfig = ElincloudServer.Common.Utils.JsonUtil.ToJson(configInfo);
                    DBServices.ElincloudConfigInstance.Update<ServiceCommonConfig>(config, p => p.ID == config.ID, ServiceCommonConfig._CommonConfig);
                }
                else
                {
                    ServiceCommonConfig config = new ServiceCommonConfig()
                    {
                        ID = DBServices.ElincloudConfigInstance.GetID<ServiceCommonConfig>(),
                        CommonConfig = ElincloudServer.Common.Utils.JsonUtil.ToJson(configInfo),
                    };
                    DBServices.ElincloudConfigInstance.Insert<ServiceCommonConfig>(config);
                }
                //更新缓存
                CommonBL.GetServiceCommonConfigInfo(true);
                Common.Utils.ConfigUtil.FileStoragePath = configInfo.FileStoragePath;

                //重启对应的服务
                CommonBL.RestartServerByConfigName(typeof(ServiceCommonConfigInfo).Name);
            }
            else
            {
                throw new ServiceException("W000009");
            }
        }

        public List<HospitalTreeNodeInfo> GetHospitalTreeNodeInfo(List<string> hospitalIDs = null)
        {

            return CommonBL.GetHospitalTreeNodeInfo(hospitalIDs);
        }
        #endregion

        #region CStore配置

        public List<CStoreConfigInfo> GetCStoreConfigInfos()
        {
            List<CStoreConfigInfo> infos = new List<CStoreConfigInfo>();
            List<CStoreConfig> configs = DBServices.ElincloudConfigInstance.GetMuch<CStoreConfig>();
            if (configs != null && configs.Count > 0)
            {
                foreach (var config in configs)
                {
                    CStoreConfigInfo info = ElincloudServer.Common.Utils.JsonUtil.FromJson<CStoreConfigInfo>(config.CStore);
                    infos.Add(info);
                }
            }
            return infos;
        }

        public CStoreConfigInfo GetCStoreConfigInfoByHospitalCode(string hospitalCode)
        {
            CStoreConfig config = DBServices.ElincloudConfigInstance.GetOne<CStoreConfig>(p => p.HospitalCode == hospitalCode);
            if (config != null)
            {
                CStoreConfigInfo info = ElincloudServer.Common.Utils.JsonUtil.FromJson<CStoreConfigInfo>(config.CStore);
                return info;
            }
            else
            {
                return new CStoreConfigInfo();
            }
        }

        public void SetCStoreConfigInfo(CStoreConfigInfo configInfo)
        {
            if (configInfo != null)
            {
                if (!string.IsNullOrEmpty(configInfo.Hospital.Code))
                {
                    //验证AE格式
                    string msg = ElincloudServer.Common.Utils.StringUtil.ValidateDicomTag_AE(configInfo.AE);
                    if (!string.IsNullOrEmpty(msg))
                    {
                        throw new ServiceException(msg);
                    }
                    CStoreConfig config = DBServices.ElincloudConfigInstance.GetOne<CStoreConfig>(p => p.HospitalCode == configInfo.Hospital.Code);
                    if (config != null)
                    {
                        config.CStore = ElincloudServer.Common.Utils.JsonUtil.ToJson(configInfo);
                        DBServices.ElincloudConfigInstance.Update<CStoreConfig>(config, p => p.HospitalCode == configInfo.Hospital.Code, CStoreConfig._CStore);
                    }
                    else
                    {
                        config = new CStoreConfig()
                        {
                            HospitalCode = configInfo.Hospital.Code,
                            CStore = ElincloudServer.Common.Utils.JsonUtil.ToJson(configInfo),
                        };
                        DBServices.ElincloudConfigInstance.Insert<CStoreConfig>(config);
                    }

                    //重启对应的服务
                    CommonBL.RestartServerByConfigName(typeof(CStoreConfigInfo).Name);
                }
                else
                {
                    throw new ServiceException("W000007");
                }
            }
            else
            {
                throw new ServiceException("W000008");
            }
        }

        #endregion

        #region Dicom设备通信配置

        public List<DicomMachineConfigInfo> GetDicomMachineInfos()
        {
            return CommonBL.GetDicomMachineInfos();
        }

        public DicomMachineConfigInfo GetDicomMachineInfoByHospitalCode(
            string hospitalCode)
        {
            return CommonBL.GetDicomMachineInfoByHospitalCode(hospitalCode);
        }

        public void SetDicomMachineInfo(DicomMachineConfigInfo configInfo)
        {
            if (configInfo == null)
            {
                throw new ServiceException("W0000013");
            }

            if (configInfo.Hospital == null || string.IsNullOrEmpty(configInfo.Hospital.Code))
            {
                throw new ServiceException("W0000012");
            }

            string hospitalCode = configInfo.Hospital.Code;
            string machines = ElincloudServer.Common.Utils.JsonUtil.ToJson(configInfo);

            DicomMachineConfig config = DBServices.ElincloudConfigInstance.GetOne<DicomMachineConfig>(p => p.HospitalCode == hospitalCode);

            if (config != null)
            {
                config.Machines = machines;

                DBServices.ElincloudConfigInstance.Update<DicomMachineConfig>(config, p => p.HospitalCode == hospitalCode, DicomMachineConfig._Machines);
            }
            else
            {
                config = new DicomMachineConfig()
                {
                    HospitalCode = hospitalCode,
                    Machines = machines,
                };

                DBServices.ElincloudConfigInstance.Insert<DicomMachineConfig>(config);
            }

            // 保存成功后，刷新当前医院的配置缓存
            CommonBL.RefreshDicomMachineConfigCache(hospitalCode);

            // 重启对应的服务
            CommonBL.RestartServerByConfigName(typeof(DicomMachineConfigInfo).Name);
        }

        #endregion

        #region 归档配置

        public List<ArchiveConfigInfo> GetArchiveConfigInfos()
        {
            List<ArchiveConfigInfo> infos = new List<ArchiveConfigInfo>();
            List<ArchiveConfig> configs = DBServices.ElincloudConfigInstance.GetMuch<ArchiveConfig>();
            if (configs != null && configs.Count > 0)
            {
                foreach (var config in configs)
                {
                    ArchiveConfigInfo info = ElincloudServer.Common.Utils.JsonUtil.FromJson<ArchiveConfigInfo>(config.Archive);
                    infos.Add(info);
                }
            }
            infos = infos.OrderBy(i => i.Hospital.ID).ToList();
            return infos;
        }

        public ArchiveConfigInfo GetArchiveConfigInfoByHospitalCode(string hospitalCode)
        {
            return CommonBL.GetArchiveConfigInfoByHospitalCode(hospitalCode);
        }

        public void SetArchiveConfigInfo(List<ArchiveConfigInfo> configInfos)
        {
            if (configInfos == null || !configInfos.Any())
                throw new ServiceException("W00000508");
            Transaction tran = DBServices.ElincloudConfigInstance.GetTransaction();
            tran.Delete<ArchiveConfig>();
            foreach (var configInfo in configInfos)
            {
                if (CheckPath(configInfo))
                {
                    ArchiveConfig config = new ArchiveConfig()
                    {
                        HospitalCode = configInfo.Hospital.Code,
                        Archive = ElincloudServer.Common.Utils.JsonUtil.ToJson(configInfo),
                    };
                    tran.Insert<ArchiveConfig>(config);
                }
            }
            if (!tran.Commit())
            {
                throw new ServiceException("W000006");
            }
            Common.Utils.ConfigUtil.ArchiveProssessCount = configInfos.FirstOrDefault().ArchiveProssessCount.ToString();//将归档的进程数存储到appsettings.json中
            foreach (var configInfo in configInfos)
            {
                //更新缓存
                CommonBL.GetArchiveConfigInfoByHospitalCode(configInfo.Hospital.Code, true);
            }
            //重启对应的服务
            CommonBL.RestartServerByConfigName(typeof(ArchiveConfigInfo).Name);
        }
        private bool CheckPath(ArchiveConfigInfo configInfo)
        {
            bool sucess = false;
            if (configInfo.IsEnableMultArchiveFolder)
            {
                var folderPaths = configInfo.ArchiveFolderConfigInfos;
                if (folderPaths == null || folderPaths.Count == 0)
                {
                    throw new ServiceException("归档路径列表为空");
                }
                foreach (var folderPath in folderPaths)
                {
                    string inputPath = folderPath.FolderName.Trim();
                    if (string.IsNullOrWhiteSpace(inputPath))
                        throw new ServiceException($"路径不能为空");
                    // 检查是否包含非法字符
                    char[] invalidChars = Path.GetInvalidPathChars();
                    if (inputPath.IndexOfAny(invalidChars) >= 0)
                    {
                        throw new ServiceException($"路径:{inputPath}包含非法字符");
                    }

                    // 检查路径长度是否超过系统限制（Windows 通常为 260 字符）
                    if (inputPath.Length > 260)
                    {
                        throw new ServiceException($"路径:{inputPath}长度超过系统限制（260字符）");
                    }

                    // 检查是否为根目录格式（如 C:\、D:\、\、/）
                    bool isRooted = Path.IsPathRooted(inputPath);
                    if (!isRooted)
                    {
                        throw new ServiceException($"路径:{inputPath}不合法！");
                    }
                    sucess = true;
                }
            }
            else
            {
                sucess = true;
            }
            return sucess;
        }

        #endregion        

        #region 数据存储服务配置

        public DataStoreConfigInfo GetDataStoreConfigInfo()
        {
            return CommonBL.GetDataStoreConfigInfo();
        }

        public void SetDataStoreConfigInfo(DataStoreConfigInfo configInfo)
        {
            if (configInfo != null)
            {
                List<DataStoreConfig> infos = DBServices.ElincloudConfigInstance.GetMuch<DataStoreConfig>();
                if (infos.Count > 0)
                {
                    DataStoreConfig config = infos.FirstOrDefault();
                    config.DataStore = ElincloudServer.Common.Utils.JsonUtil.ToJson(configInfo);
                    DBServices.ElincloudConfigInstance.Update<DataStoreConfig>(config, p => p.ID == config.ID, DataStoreConfig._DataStore);
                }
                else
                {
                    DataStoreConfig config = new DataStoreConfig()
                    {
                        ID = DBServices.ElincloudConfigInstance.GetID<DataStoreConfig>(),
                        DataStore = ElincloudServer.Common.Utils.JsonUtil.ToJson(configInfo),
                    };
                    DBServices.ElincloudConfigInstance.Insert<DataStoreConfig>(config);
                }
                //更新缓存
                CommonBL.GetDataStoreConfigInfo(true);

                //重启对应的服务
                CommonBL.RestartServerByConfigName(typeof(DataStoreConfigInfo).Name);
            }
            else
            {
                throw new ServiceException("W0000029");
            }
        }

        #endregion

        #region 数据库备份服务
        public DBBackupConfigInfo GetDBBackupConfigInfo()
        {
            return CommonBL.GetDBBackupConfigInfo();
        }

        public void SetDBBackupConfigInfo(DBBackupConfigInfo configInfo)
        {
            if (configInfo != null)
            {
                List<DBBackupConfig> infos = DBServices.ElincloudConfigInstance.GetMuch<DBBackupConfig>();
                if (infos.Count > 0)
                {
                    DBBackupConfig config = infos.FirstOrDefault();
                    config.DBBackup = ElincloudServer.Common.Utils.JsonUtil.ToJson(configInfo);
                    DBServices.ElincloudConfigInstance.Update<DBBackupConfig>(config, p => p.ID == config.ID, DBBackupConfig._DBBackup);
                }
                else
                {
                    DBBackupConfig config = new DBBackupConfig()
                    {
                        ID = DBServices.ElincloudConfigInstance.GetID<DBBackupConfig>(),
                        DBBackup = ElincloudServer.Common.Utils.JsonUtil.ToJson(configInfo),
                    };
                    DBServices.ElincloudConfigInstance.Insert<DBBackupConfig>(config);
                }

                //重启对应的服务
                CommonBL.RestartServerByConfigName(typeof(DBBackupConfigInfo).Name);
            }
            else
            {
                throw new ServiceException("W0000034");
            }
        }

        /// <summary>
        /// 获取数据库名称列表
        /// </summary>
        /// <returns></returns>
        public List<string> GetDBNames()
        {
            List<string> ret = new List<string>();
            List<string> dbs = DBServices.ElincloudBasicInstance.GetDBNames();
            if (dbs != null && dbs.Count > 0)
            {
                ret.AddRange(dbs);
            }
            return ret;
        }
        #endregion

        #region 清除文件配置
        /// <summary>
        /// 返回缓存子文件夹目录
        /// </summary>
        /// <returns></returns>
        public List<KeyValueModel> GetCacheSubfolders()
        {
            return ApplicationEx.CacheSubfolder;
        }

        public ClearFileConfigInfo GetClearFileConfigInfo()
        {
            return CommonBL.GetClearFileConfigInfo();
        }

        public void SetClearConfigInfo(ClearFileConfigInfo configInfo)
        {
            if (configInfo != null)
            {
                List<ClearFileConfig> infos = DBServices.ElincloudConfigInstance.GetMuch<ClearFileConfig>();
                if (infos.Count > 0)
                {
                    ClearFileConfig config = infos.FirstOrDefault();
                    config.ClearFile = ElincloudServer.Common.Utils.JsonUtil.ToJson(configInfo);
                    DBServices.ElincloudConfigInstance.Update<ClearFileConfig>(config, p => p.ID == config.ID, ClearFileConfig._ClearFile);
                }
                else
                {
                    ClearFileConfig config = new ClearFileConfig()
                    {
                        ID = DBServices.ElincloudConfigInstance.GetID<ClearFileConfig>(),
                        ClearFile = ElincloudServer.Common.Utils.JsonUtil.ToJson(configInfo),
                    };
                    DBServices.ElincloudConfigInstance.Insert<ClearFileConfig>(config);
                }

                //重启对应的服务
                CommonBL.RestartServerByConfigName(typeof(ClearFileConfigInfo).Name);
            }
            else
            {
                throw new ServiceException("W0000030");
            }
        }

        #endregion

        #region 影像工作站配置

        #region 四角信息
        public WorkFilmFourCornerConfigInfo? GetWorkFilmFourCornerConfigInfoByLoginName(string loginName)
        {
            WorkFilmFourCornerConfig config = DBServices.ElincloudConfigInstance.GetOne<WorkFilmFourCornerConfig>(p => p.LoginName == loginName);
            if (config != null)
            {
                WorkFilmFourCornerConfigInfo info = ElincloudServer.Common.Utils.JsonUtil.FromJson<WorkFilmFourCornerConfigInfo>(config.FilmFourCorner);
                return info;
            }
            return null;
        }

        public void SetWorkFilmFourCornerConfigInfo(WorkFilmFourCornerConfigInfo configInfo)
        {
            if (configInfo != null)
            {
                if (!string.IsNullOrEmpty(configInfo.LoginName))
                {
                    WorkFilmFourCornerConfig config = DBServices.ElincloudConfigInstance.GetOne<WorkFilmFourCornerConfig>(p => p.LoginName == configInfo.LoginName);
                    if (config != null)
                    {
                        config.FilmFourCorner = ElincloudServer.Common.Utils.JsonUtil.ToJson(configInfo);
                        DBServices.ElincloudConfigInstance.Update<WorkFilmFourCornerConfig>(config, p => p.LoginName == configInfo.LoginName, WorkFilmFourCornerConfig._FilmFourCorner);
                    }
                    else
                    {
                        config = new WorkFilmFourCornerConfig()
                        {
                            LoginName = configInfo.LoginName,
                            FilmFourCorner = ElincloudServer.Common.Utils.JsonUtil.ToJson(configInfo),
                        };
                        DBServices.ElincloudConfigInstance.Insert<WorkFilmFourCornerConfig>(config);
                    }
                }
                else
                {
                    throw new ServiceException("W0000057");
                }
            }
            else
            {
                throw new ServiceException("W0000058");
            }
        }
        #endregion

        #region 通用设置
        public WorkFilmSettingsConfigInfo? GetWorkFilmSettingsConfigInfoByLoginName(string loginName)
        {
            WorkFilmSettingsConfig config = DBServices.ElincloudConfigInstance.GetOne<WorkFilmSettingsConfig>(p => p.LoginName == loginName);
            if (config != null)
            {
                WorkFilmSettingsConfigInfo info = ElincloudServer.Common.Utils.JsonUtil.FromJson<WorkFilmSettingsConfigInfo>(config.FilmSettings);
                return info;
            }
            return null;
        }

        public void SetWorkFilmSettingsConfigInfo(WorkFilmSettingsConfigInfo configInfo)
        {
            if (configInfo != null)
            {
                if (!string.IsNullOrEmpty(configInfo.LoginName))
                {
                    WorkFilmSettingsConfig config = DBServices.ElincloudConfigInstance.GetOne<WorkFilmSettingsConfig>(p => p.LoginName == configInfo.LoginName);
                    if (config != null)
                    {
                        config.FilmSettings = ElincloudServer.Common.Utils.JsonUtil.ToJson(configInfo);
                        DBServices.ElincloudConfigInstance.Update<WorkFilmSettingsConfig>(config, p => p.LoginName == configInfo.LoginName, WorkFilmSettingsConfig._FilmSettings);
                    }
                    else
                    {
                        config = new WorkFilmSettingsConfig()
                        {
                            LoginName = configInfo.LoginName,
                            FilmSettings = ElincloudServer.Common.Utils.JsonUtil.ToJson(configInfo),
                        };
                        DBServices.ElincloudConfigInstance.Insert<WorkFilmSettingsConfig>(config);
                    }
                }
                else
                {
                    throw new ServiceException("W0000059");
                }
            }
            else
            {
                throw new ServiceException("W0000060");
            }
        }
        #endregion

        #endregion

        #region 密码强度验证配置
        public PasswordPolicyConfigInfo GetPasswordPolicyConfigInfoByHospitalCode(string hospitalCode)
        {
            return CommonBL.GetPasswordPolicyConfigInfoByHospitalCode(hospitalCode);
        }

        public void SetPasswordPolicyConfigInfo(PasswordPolicyConfigInfo configInfo)
        {
            if (configInfo != null)
            {
                if (!string.IsNullOrEmpty(configInfo.HospitalCode))
                {
                    PasswordPolicyConfig config = DBServices.ElincloudConfigInstance.GetOne<PasswordPolicyConfig>(p => p.HospitalCode == configInfo.HospitalCode);
                    if (config != null)
                    {
                        config.PasswordPolicy = ElincloudServer.Common.Utils.JsonUtil.ToJson(configInfo);
                        DBServices.ElincloudConfigInstance.Update<PasswordPolicyConfig>(config, p => p.HospitalCode == configInfo.HospitalCode, PasswordPolicyConfig._PasswordPolicy);
                    }
                    else
                    {
                        config = new PasswordPolicyConfig()
                        {
                            HospitalCode = configInfo.HospitalCode,
                            PasswordPolicy = ElincloudServer.Common.Utils.JsonUtil.ToJson(configInfo),
                        };
                        DBServices.ElincloudConfigInstance.Insert<PasswordPolicyConfig>(config);
                    }
                }
                else
                {
                    throw new ServiceException("W0000024");
                }
            }
            else
            {
                throw new ServiceException("W0000025");
            }
        }
        #endregion

        #region 返回文件访问地址
        public string GetFileSeriviceUrl(string accessIP)
        {
            ServiceCommonConfigInfo folderConfig = CommonBL.GetServiceCommonConfigInfo();
            ServerAccessItem item = folderConfig.ServerAccessItemList.FirstOrDefault(i => i.AccessIP == accessIP);
            if (item == null)
            {
                item = folderConfig.GetDefaultItem();
            }
            string result = string.Empty;
            if (item != null)
            {
                result = item.ServerAddress;
            }
            return result;
        }
        #endregion

        #region 返回影像工作站访问地址
        public string GetDicomViewerSeriviceUrl(string accessIP)
        {
            ServiceCommonConfigInfo folderConfig = CommonBL.GetServiceCommonConfigInfo();
            ServerAccessItem item = folderConfig.ServerAccessItemList.FirstOrDefault(i => i.AccessIP == accessIP);
            if (item == null)
            {
                item = folderConfig.GetDefaultItem();
            }
            return item.DicomViewAddress;
        }
        #endregion
    }
}
