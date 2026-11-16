using ElincloudServer.Common;
using ElincloudServer.DBEntity;
using ElincloudServer.DBService;
using ElincloudServer.Entity;

namespace ElincloudServer.CommonBLService
{

    public class DBServices
    {

        #region 医院数据库

        private static List<DBInstance> hospitalDBInstancees = new List<DBInstance>();
        public static SqlAccess GetHospitalDBInstance(string hospitalGroupID)
        {
            try
            {
                lock (hospitalDBInstancees)
                {
                    if (!string.IsNullOrEmpty(hospitalGroupID))
                    {
                        DBInstance dbinstance = hospitalDBInstancees.FirstOrDefault(p => p.Key == hospitalGroupID);
                        if (dbinstance != null)
                        {
                            if (dbinstance.Instance == null)
                            {
                                dbinstance.Instance = new SqlAccess(hospitalGroupID);
                            }
                        }
                        else
                        {
                            dbinstance = new DBInstance()
                            {
                                Key = hospitalGroupID,
                                Instance = new SqlAccess(hospitalGroupID),
                            };
                            hospitalDBInstancees.Add(dbinstance);
                        }
                        return dbinstance.Instance;
                    }
                    else
                    {
                        throw new ServiceException("W00000330");
                    }
                }
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog(ex.Message, ex);
                throw new Exception("W00000331");
            }
        }

        #endregion

        #region 基础本地数据库

        private static SqlAccess elincloudBasicInstance;
        public static SqlAccess ElincloudBasicInstance
        {
            get
            {
                if (elincloudBasicInstance == null)
                {
                    try
                    {
                        elincloudBasicInstance = new SqlAccess(LocalDBNames.ElincloudBasic.ToString());
                    }
                    catch (Exception ex)
                    {
                        ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("[初始化ElincloudBasic数据库连接异常] " + ex.Message, ex);
                        throw new ServiceException("W00000332");
                    }
                }
                return elincloudBasicInstance;
            }
        }

        private static SqlAccess elincloudLogInstance;
        public static SqlAccess ElincloudLogInstance
        {
            get
            {
                if (elincloudLogInstance == null)
                {
                    try
                    {
                        elincloudLogInstance = new SqlAccess(LocalDBNames.ElincloudLog.ToString());
                    }
                    catch (Exception ex)
                    {
                        ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("[初始化ElincloudLog数据库连接异常] " + ex.Message, ex);
                        throw new ServiceException("W00000333");
                    }
                }
                return elincloudLogInstance;
            }
        }

        private static SqlAccess elincloudConfigInstance;
        public static SqlAccess ElincloudConfigInstance
        {
            get
            {
                if (elincloudConfigInstance == null)
                {
                    try
                    {
                        elincloudConfigInstance = new SqlAccess(LocalDBNames.ElincloudConfig.ToString());
                    }
                    catch (Exception ex)
                    {
                        ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("[初始化ElincloudConfig数据库连接异常] " + ex.Message, ex);
                        throw new ServiceException("W00000334");
                    }
                }
                return elincloudConfigInstance;
            }
        }

        private static SqlAccess elincloudThirdPartInstance;
        public static SqlAccess ElincloudThirdPartInstance
        {
            get
            {
                if (elincloudThirdPartInstance == null)
                {
                    try
                    {
                        elincloudThirdPartInstance = new SqlAccess(LocalDBNames.ElincloudThirdPart.ToString());
                    }
                    catch (Exception ex)
                    {
                        ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("[初始化ElincloudThirdPart数据库连接异常] " + ex.Message, ex);
                        throw new ServiceException("W00000335");
                    }
                }
                return elincloudThirdPartInstance;
            }
        }

        private static SqlAccess elincloudForwardInstance;
        public static SqlAccess ElincloudForwardInstance
        {
            get
            {
                if (elincloudForwardInstance == null)
                {
                    try
                    {
                        elincloudForwardInstance = new SqlAccess(LocalDBNames.ForwardingSystemDB.ToString());
                    }
                    catch (Exception ex)
                    {
                        ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("[初始化ForwardingSystemDB数据库连接异常] " + ex.Message, ex);
                        throw new ServiceException("初始化ForwardingSystemDB数据库连接异常!");
                    }
                }
                return elincloudForwardInstance;
            }
        }

        private static SqlAccess elincloudForwardConfigInstance;
        public static SqlAccess ElincloudForwardConfigInstance
        {
            get
            {
                if (elincloudForwardConfigInstance == null)
                {
                    try
                    {
                        elincloudForwardConfigInstance = new SqlAccess(LocalDBNames.ForwardingSystemConfigDB.ToString());
                    }
                    catch (Exception ex)
                    {
                        ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("[初始化ForwardingSystemConfigDB数据库连接异常] " + ex.Message, ex);
                        throw new ServiceException("初始化ForwardingSystemConfigDB数据库连接异常!");
                    }
                }
                return elincloudForwardConfigInstance;
            }
        }

        private static SqlAccess elincloudForwardLogInstance;
        public static SqlAccess ElincloudForwardLogInstance
        {
            get
            {
                if (elincloudForwardLogInstance == null)
                {
                    try
                    {
                        elincloudForwardLogInstance = new SqlAccess(LocalDBNames.ForwardingSystemLogDB.ToString());
                    }
                    catch (Exception ex)
                    {
                        ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("[初始化ForwardingSystemLogDB数据库连接异常] " + ex.Message, ex);
                        throw new ServiceException("初始化ForwardingSystemLogDB数据库连接异常!");
                    }
                }
                return elincloudForwardLogInstance;
            }
        }

        #endregion

        #region 第三方系统数据库

        private static List<DBInstance> thirdPartDBInstances = new List<DBInstance>();

        private static DBConnectionInfo getThirdPartDBConfig(ThirdPartSystemTypes systemType)
        {
            ThirdPartSystemConfig config = ElincloudConfigInstance.GetOne<ThirdPartSystemConfig>(p => p.ThirdPartSystemType == (int)systemType);
            if (config != null)
            {
                ThirdPartSystemConfigInfo info = ElincloudServer.Common.Utils.JsonUtil.FromJson<ThirdPartSystemConfigInfo>(config.ThirdPart);
                DBConnectionInfo dbConfigInfo = new DBConnectionInfo()
                {
                    DBType = info.DBType,
                    IP = info.IP,
                    Port = info.Port,
                    UserName = info.UserName,
                    Password = ElincloudServer.Common.Utils.StringUtil.Decrypt(info.Password),
                    DBName = info.DBName
                };
                return dbConfigInfo;
            }
            else
            {
                throw new ServiceException("W00000338");
            }
        }


        public static SqlAccess GetThirdPartDBInstance(ThirdPartSystemTypes systemType)
        {
            try
            {
                lock (thirdPartDBInstances)
                {
                    DBInstance dbinstance = thirdPartDBInstances.FirstOrDefault(p => p.Key == systemType.ToString());
                    if (dbinstance != null)
                    {
                        if (dbinstance.Instance == null)
                        {
                            DBConnectionInfo conn = getThirdPartDBConfig(systemType);
                            dbinstance.Instance = new SqlAccess(conn);
                        }
                    }
                    else
                    {
                        DBConnectionInfo conn = getThirdPartDBConfig(systemType);
                        dbinstance = new DBInstance()
                        {
                            Key = systemType.ToString(),
                            Instance = new SqlAccess(conn),
                        };
                        thirdPartDBInstances.Add(dbinstance);
                    }
                    return dbinstance.Instance;

                }
            }
            catch (ServiceException ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog(ex.Message, ex);
                return null;
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("[初始化数据库连接异常] " + ex.Message, ex);
                throw new ServiceException("W00000331");
            }
        }


        public static void SetThirdPartDBInstance(ThirdPartSystemConfigInfo configInfo)
        {
            int dbIndex = thirdPartDBInstances.FindIndex(p => p.Key == configInfo.ThirdPartSystemType.ToString());
            if (dbIndex > -1)
            {
                DBConnectionInfo conn = getThirdPartDBConfig(configInfo.ThirdPartSystemType);

                DBInstance dbinstance = new DBInstance()
                {
                    Key = configInfo.ThirdPartSystemType.ToString(),
                    Instance = new SqlAccess(conn),
                };
                thirdPartDBInstances[dbIndex] = dbinstance;
            }
        }

        public static SqlAccess GetReportQRCodeDBConfig(string hospitalCode)
        {
            DBInstance dbinstance = new DBInstance()
            {
                Key = hospitalCode,
                Instance = new SqlAccess(getReportQRCodeDBConfig(hospitalCode)),
            };
            return dbinstance.Instance;
        }
        private static DBConnectionInfo getReportQRCodeDBConfig(string hospitalCode)
        {
            ReportQRCodeConfig config = DBServices.ElincloudConfigInstance.GetOne<ReportQRCodeConfig>(p => p.HospitalCode == hospitalCode);
            if (config != null)
            {
                DBConnectionInfo info = ElincloudServer.Common.Utils.JsonUtil.FromJson<DBConnectionInfo>(config.ReportQRCode);
                return info;
            }
            else
            {
                throw new ServiceException("W00000337");
            }
        }


        #endregion

    }

}
