using Dm;
using ElincloudAISliceServer.DBEntity;
using ElincloudAISliceServer.Entity;
using ElincloudServer.Common;
using ElincloudServer.CommonBLService;
using ElincloudServer.DBEntity;
using ElincloudServer.DBService;
using ElincloudServer.Entity;
using Google.Protobuf.WellKnownTypes;
using Microsoft.AspNetCore.Mvc.ViewFeatures;
using System.Collections.Generic;
using System.Data;
using System.Data.SqlClient;
using System.Diagnostics;
using System.Drawing.Printing;
using System.Linq;
using System.Linq.Expressions;
using System.Text;
using System.Text.RegularExpressions;
using static Amazon.S3.Util.S3EventNotification;
using ExaminationImage = ElincloudAISliceServer.DBEntity.ExaminationImage;
using RISExamination = ElincloudAISliceServer.DBEntity.RISExamination;
using DCMImageInfo = ElincloudAISliceServer.Entity.DCMImageInfo;

namespace ElincloudServer.SystemBLService
{
    public class SystemBL
    {
        #region 医院管理

        public List<HospitalInfo> GetHospitals()
        {
            return CommonBL.GetHospitalInfos();
        }
        public Page<HospitalInfo> GetHospitalInfosByPage(int pageIndex, int pageSize)
        {
            List<HospitalInfo> rets = new List<HospitalInfo>();
            if (pageSize > 0 && pageIndex > 0)
            {
                Expression<Func<Hospital, bool>> whereExpression = null;

                Page<Hospital> hospitals = DBServices.ElincloudBasicInstance.GetMuchByPage<Hospital>(new DBPage(pageSize, pageIndex), whereExpression);
                foreach (var hospital in hospitals.Content)
                {
                    rets.Add(new HospitalInfo()
                    {
                        ID = hospital.ID,
                        Name = hospital.Name,
                        Remark = hospital.Remark,
                        Code = hospital.Code,
                        EasyName = hospital.EasyName,
                        SortNumber = hospital.SortNumber,
                        Address = hospital.Address,
                        Longitude = hospital.Longitude,
                        Latitude = hospital.Latitude,
                        HospitalGroupID = hospital.HospitalGroupID,
                        HospitalGroupName = hospital.HospitalGroupName,
                        AreaID = hospital.AreaID,
                        AreaName = hospital.AreaName,
                        CityID = hospital.CityID,
                        CityName = hospital.CityName,
                        HospitalFreeTypes = hospital.HospitalFreeTypes,
                        HospitalRegionTypes = hospital.HospitalRegionTypes,
                        HospitalSizes = hospital.HospitalSizes,
                        ProvinceID = hospital.ProvinceID,
                        ProvinceName = hospital.ProvinceName
                    });
                }
                return new Page<HospitalInfo>(pageSize, pageIndex, hospitals.TotalElements, rets);
            }
            return new Page<HospitalInfo>(pageSize, pageIndex, 0, rets);
        }

        public string AddHospital(HospitalInfo info)
        {
            if (info != null)
            {
                Hospital hospitalInfo = DBServices.ElincloudBasicInstance.GetOne<Hospital>(x => x.Code == info.Code);

                if (hospitalInfo != null)
                {
                    throw new ServiceException("W00000187");
                }
                int count = DBServices.ElincloudBasicInstance.GetCount<Hospital>(p => p.Name == info.Name && p.Code != info.Code);
                if (count > 0)
                {
                    throw new ServiceException("W00000494");
                }
                Hospital hospital = new Hospital()
                {
                    ID = DBServices.ElincloudBasicInstance.GetID<Hospital>(4),
                    Code = info.Code,
                    Name = info.Name,
                    EasyName = info.EasyName,
                    Remark = info.Remark,
                    SortNumber = (int)info.SortNumber,
                    Longitude = info.Longitude,
                    Latitude = info.Latitude,
                    Address = info.Address,
                    HospitalGroupID = string.IsNullOrWhiteSpace(info.HospitalGroupID) ? DBServices.ElincloudBasicInstance.GetID<Hospital>(4, "H", Hospital._HospitalGroupID) : info.HospitalGroupID,
                    HospitalGroupName = info.HospitalGroupName,
                    CreateDate = DateTime.Now,
                    UpdateDate = DateTime.Now,
                    AreaID = info.AreaID,
                    AreaName = info.AreaName,
                    ProvinceName = info.ProvinceName,
                    ProvinceID = info.ProvinceID,
                    HospitalSizes = info.HospitalSizes,
                    HospitalRegionTypes = info.HospitalRegionTypes,
                    HospitalFreeTypes = info.HospitalFreeTypes,
                    CityName = info.CityName,
                    CityID = info.CityID
                };

                //创建数据库时此脚本执行时间过长，所以此处增加线程，由前端定时获取库中数据来判断库是否创建成功
                Common.Utils.ThreadUtil.StartThread(new
                Action(() =>
                {
                    string DBIP = Common.Utils.ConfigUtil.IP;
                    string DBPort = Common.Utils.ConfigUtil.Port;
                    string DBUserName = Common.Utils.ConfigUtil.UserName;
                    string DBPassword = Common.Utils.ConfigUtil.Password;
                    string dBName = LocalDBNames.ElincloudBasic.ToString();
                    if (DBTypes.SqlServer.ToString() == Utils.ConfigUtil.DBType)
                    {
                        string dataSource = DBIP.Contains("\\") ? DBIP : string.Format("{0},{1}", DBIP, DBPort);
                        string connectionStr = string.Format("Data Source={0};Initial Catalog={1};User ID={2};Password={3};TrustServerCertificate=true", dataSource, dBName, DBUserName, DBPassword);
                        using (SqlConnection conn = new SqlConnection(connectionStr))
                        {
                            using (SqlCommand cmd = new SqlCommand(string.Format("exec {0} '{1}';", ElincloudServer.Common.ApplicationEx.CreateHospitalProcName, hospital.HospitalGroupID), conn))
                            {
                                try
                                {
                                    conn.Open();
                                    cmd.CommandTimeout = 300;
                                    cmd.CommandType = CommandType.Text;
                                    cmd.ExecuteNonQuery();
                                    DBServices.ElincloudBasicInstance.Insert<Hospital>(hospital);

                                    using (SqlCommand cmd2 = new SqlCommand(string.Format("exec {0} '{1}', '{2}';", ElincloudServer.Common.ApplicationEx.CreateBasicDataProcName, hospital.HospitalGroupID, hospital.ID), conn))
                                    {
                                        cmd2.CommandTimeout = 300;
                                        cmd2.CommandType = CommandType.Text;
                                        cmd2.ExecuteNonQuery();
                                    }
                                    //重启对应的服务
                                    CommonBL.RestartServerByConfigName(typeof(HospitalInfo).Name);
                                }
                                catch (Exception ex)
                                {
                                    ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("创建数据库失败!", ex);
                                }
                            }
                        }
                        //DBServices.ElincloudBasicInstance.ExecuteProcedure(ElincloudServer.Common.ApplicationEx.CreateHospitalProcName, hospital.HospitalGroupID);
                    }
                    else if (DBTypes.DM.ToString() == Utils.ConfigUtil.DBType)
                    {
                        string connectionStr = string.Format("Server={0};Port={1};Database={2};User ID={3};Password={4};", DBIP, DBPort, dBName, DBUserName, DBPassword);
                        using (DmConnection conn = new DmConnection(connectionStr))
                        {
                            using (DmCommand cmd = new DmCommand(string.Format("call {0}.{1} ('{2}');", dBName, ElincloudServer.Common.ApplicationEx.CreateHospitalProcName, hospital.HospitalGroupID), conn))
                            {
                                try
                                {
                                    conn.Open();
                                    cmd.CommandTimeout = 300;
                                    cmd.CommandType = CommandType.Text;
                                    cmd.ExecuteNonQuery();
                                    DBServices.ElincloudBasicInstance.Insert<Hospital>(hospital);
                                    //重启对应的服务
                                    CommonBL.RestartServerByConfigName(typeof(HospitalInfo).Name);
                                }
                                catch (Exception ex)
                                {
                                    ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("创建数据库失败!", ex);
                                }
                            }
                        }
                    }
                }));
                return hospital.ID;
            }
            else
            {
                throw new ServiceException("W00000188");
            }
        }

        public void UpdateHospital(HospitalInfo info)
        {
            if (info != null)
            {
                Hospital hospital = DBServices.ElincloudBasicInstance.GetOne<Hospital>(p => p.ID == info.ID);
                if (hospital != null)
                {
                    int count = DBServices.ElincloudBasicInstance.GetCount<Hospital>(p => p.Name == info.Name && p.ID != info.ID);
                    if (count > 0)
                    {
                        throw new ServiceException("W00000494");
                    }
                    hospital.Name = info.Name;
                    hospital.Code = info.Code;
                    hospital.Remark = info.Remark;
                    hospital.EasyName = info.EasyName;
                    hospital.SortNumber = info.SortNumber;
                    hospital.Longitude = info.Longitude;
                    hospital.Latitude = info.Latitude;
                    hospital.Address = info.Address;
                    hospital.HospitalGroupID = info.HospitalGroupID;
                    hospital.HospitalGroupName = info.HospitalGroupName;
                    hospital.AreaID = info.AreaID;
                    hospital.AreaName = info.AreaName;
                    hospital.ProvinceName = info.ProvinceName;
                    hospital.ProvinceID = info.ProvinceID;
                    hospital.HospitalSizes = info.HospitalSizes;
                    hospital.HospitalRegionTypes = info.HospitalRegionTypes;
                    hospital.HospitalFreeTypes = info.HospitalFreeTypes;
                    hospital.CityName = info.CityName;
                    hospital.CityID = info.CityID;
                    hospital.UpdateDate = DateTime.Now;
                    DBServices.ElincloudBasicInstance.Update<Hospital>(hospital, p => p.ID == info.ID,
                        Hospital._Name,
                        Hospital._Code,
                        Hospital._Remark,
                        Hospital._EasyName,
                        Hospital._SortNumber,
                        Hospital._Longitude,
                        Hospital._Latitude,
                        Hospital._Address,
                        Hospital._HospitalGroupID,
                        Hospital._HospitalGroupName,
                        Hospital._UpdateDate,
                        Hospital._AreaID,
                        Hospital._AreaName,
                        Hospital._CityID,
                        Hospital._CityName,
                        Hospital._HospitalSizes,
                        Hospital._HospitalFreeTypes,
                        Hospital._HospitalRegionTypes,
                        Hospital._ProvinceID,
                        Hospital._ProvinceName
                        );
                }
                else
                {
                    throw new ServiceException("W00000190");
                }

            }
            else
            {
                throw new ServiceException("W00000189");
            }
        }

        #endregion

        #region 医院授权

        /// <summary>
        /// 获取医院授权集合
        /// </summary>
        /// <returns></returns>
        public List<HospitalRegistInfo> GetHospitalRegisterInfos()
        {
            List<HospitalRegistInfo> ret = new List<HospitalRegistInfo>();
            List<HospitalRegist> list = DBServices.ElincloudBasicInstance.GetMuch<HospitalRegist>();
            if (list != null)
            {
                List<Hospital> hospitals = DBServices.ElincloudBasicInstance.GetMuch<Hospital>();
                foreach (var item in list)
                {
                    Hospital hospital = hospitals.FirstOrDefault(p => p.ID == item.HospitalID);
                    if (hospital != null)
                    {
                        ret.Add(new HospitalRegistInfo()
                        {
                            HospitalID = hospital.ID,
                            HospitalName = hospital.Name,
                            RegistKey = item.RegistKey,
                            RegistCode = item.RegistCode,
                            HospitalStatus = ElincloudServer.Common.Utils.EnumUtil.GetEnumValue<CloudImageHospitalStatus>(item.HospitalStatus),
                            LastConnectDate = item.LastConnectDate,
                            Version = item.Version,
                            PermissionRemainDays = ElincloudServer.Common.ApplicationEx.GetRemainDays(item.RegistCode, item.RegistKey),
                        });
                    }
                }
            }
            return ret;
        }

        /// <summary>
        /// 添加医院授权信息
        /// </summary>
        /// <param name="info"></param>
        /// <exception cref="ServiceException"></exception>
        public void AddHospitalRegisterInfo(HospitalRegistInfo info)
        {
            if (info != null)
            {
                if (string.IsNullOrWhiteSpace(info.RegistCode))
                {
                    HospitalRegist hr = DBServices.ElincloudBasicInstance.GetOne<HospitalRegist>(p => p.HospitalID == info.HospitalID);
                    if (hr == null)
                    {
                        hr = new HospitalRegist()
                        {
                            HospitalID = info.HospitalID,
                            RegistCode = info.RegistCode,
                            RegistKey = info.RegistKey,
                            HospitalStatus = (int)info.HospitalStatus,
                            LastConnectDate = new DateTime(1900, 1, 1),
                            Version = "0.0.0.0"
                        };
                        DBServices.ElincloudBasicInstance.Insert<HospitalRegist>(hr);
                    }
                    else
                    {
                        throw new ServiceException("W00000309");
                    }
                }
                else
                {
                    int remainDays = ElincloudServer.Common.ApplicationEx.GetRemainDays(info.RegistCode, info.RegistKey);
                    if (remainDays >= 0)
                    {
                        HospitalRegist hr = DBServices.ElincloudBasicInstance.GetOne<HospitalRegist>(p => p.HospitalID == info.HospitalID);
                        if (hr == null)
                        {
                            hr = new HospitalRegist()
                            {
                                HospitalID = info.HospitalID,
                                RegistCode = info.RegistCode,
                                RegistKey = info.RegistKey,
                                HospitalStatus = (int)info.HospitalStatus,
                                LastConnectDate = new DateTime(1900, 1, 1),
                                Version = "0.0.0.0"
                            };
                            DBServices.ElincloudBasicInstance.Insert<HospitalRegist>(hr);
                        }
                        else
                        {
                            throw new ServiceException("W00000309");
                        }
                    }
                    else
                    {
                        throw new ServiceException("W00000310");
                    }
                }
            }
            else
            {
                throw new ServiceException("W00000311");
            }
        }

        /// <summary>
        /// 修改医院授权信息
        /// </summary>
        /// <param name="info"></param>
        /// <exception cref="ServiceException"></exception>
        public void UpdateHospitalRegisterInfo(HospitalRegistInfo info)
        {
            if (info != null)
            {
                if (string.IsNullOrWhiteSpace(info.RegistCode))
                {
                    HospitalRegist hr = DBServices.ElincloudBasicInstance.GetOne<HospitalRegist>(p => p.HospitalID == info.HospitalID);
                    if (hr != null)
                    {
                        hr.RegistCode = info.RegistCode;
                        hr.RegistKey = info.RegistKey;
                        hr.HospitalStatus = (int)info.HospitalStatus;
                        DBServices.ElincloudBasicInstance.Update<HospitalRegist>(hr, p => p.HospitalID == hr.HospitalID, HospitalRegist._RegistKey, HospitalRegist._RegistCode, HospitalRegist._HospitalStatus);
                    }
                    else
                    {
                        throw new ServiceException("W00000312");
                    }
                }
                else
                {
                    int remainDays = ElincloudServer.Common.ApplicationEx.GetRemainDays(info.RegistCode, info.RegistKey);
                    if (remainDays >= 0)
                    {
                        HospitalRegist hr = DBServices.ElincloudBasicInstance.GetOne<HospitalRegist>(p => p.HospitalID == info.HospitalID);
                        if (hr != null)
                        {
                            DateTime oldDt = DateTime.Now.AddYears(-10);
                            if (!string.IsNullOrWhiteSpace(hr.RegistCode))
                            {
                                string oldDecriptKey = ElincloudServer.Common.Utils.StringUtil.Decrypt(hr.RegistCode, hr.RegistKey);
                                string[] oldkeys = oldDecriptKey.Split('|');
                                oldDt = DateTime.ParseExact(oldkeys[0], "yyyyMMddHH", System.Globalization.CultureInfo.CurrentCulture);
                            }

                            string newDecriptKey = ElincloudServer.Common.Utils.StringUtil.Decrypt(info.RegistCode, info.RegistKey);
                            string[] newkeys = newDecriptKey.Split('|');
                            DateTime newDt = DateTime.ParseExact(newkeys[0], "yyyyMMddHH", System.Globalization.CultureInfo.CurrentCulture);
                            if (newDt >= oldDt)
                            {
                                hr.RegistCode = info.RegistCode;
                                hr.RegistKey = info.RegistKey;
                                hr.HospitalStatus = (int)info.HospitalStatus;
                                DBServices.ElincloudBasicInstance.Update<HospitalRegist>(hr, p => p.HospitalID == hr.HospitalID, HospitalRegist._RegistKey, HospitalRegist._RegistCode, HospitalRegist._HospitalStatus);
                            }
                            else
                            {
                                throw new ServiceException("W00000479");
                            }
                        }
                        else
                        {
                            throw new ServiceException("W00000312");
                        }
                    }
                    else
                    {
                        throw new ServiceException("W00000310");
                    }
                }
            }
            else
            {
                throw new ServiceException("W00000311");
            }
        }

        /// <summary>
        /// 删除医院授权信息
        /// </summary>
        /// <param name="info"></param>
        /// <exception cref="ServiceException"></exception>
        public void DeleteHospitalRegisterInfoByHospitalID(string hospitalID)
        {
            DBServices.ElincloudBasicInstance.Delete<HospitalRegist>(p => p.HospitalID == hospitalID);
        }

        #endregion

        #region 日志查询

        #region SQL日志

        public Page<SQLLogInfo> GetSQLLogInfosByPage(DateTime startDate, DateTime endDate, string searchString, string hospitalNumber, int pageIndex, int pageSize)
        {

            List<SQLLogInfo> logInfos = new List<SQLLogInfo>();
            if (searchString != string.Empty)
            {
                searchString = ElincloudServer.Common.Utils.StringUtil.Decrypt(searchString);

            }
            System.Linq.Expressions.Expression<Func<SQLLog, bool>> whereExpression = p => p.CreateDate >= startDate && p.CreateDate <= endDate;

            if (!string.IsNullOrEmpty(searchString))
            {
                whereExpression = whereExpression.And(p => p.SQLContent.Contains(searchString));
            }
            if (!string.IsNullOrEmpty(hospitalNumber))
            {
                whereExpression = whereExpression.And(p => p.HospitalNumber == hospitalNumber);
            }

            Page<SQLLog> pageLogs = DBServices.ElincloudLogInstance.GetMuchByPage<SQLLog>(new DBPage(pageSize, pageIndex), whereExpression, query => query.OrderByDescending(x => x.CreateDate));

            foreach (var log in pageLogs.Content)
            {
                SQLLogInfo logInfo = new SQLLogInfo()
                {
                    CreateDate = log.CreateDate,
                    SQLContent = log.SQLContent,
                };
                logInfos.Add(logInfo);
            }

            return new Entity.Page<SQLLogInfo>(pageSize, pageIndex, pageLogs.TotalElements, logInfos);
        }

        #endregion

        #region 审计日志

        public Page<AuditLogInfo> GetAuditLogInfosByPage(DateTime startDate, DateTime endDate, string searchString, string hospitalNumber, int pageIndex, int pageSize)
        {
            List<AuditLogInfo> logInfos = new List<AuditLogInfo>();

            System.Linq.Expressions.Expression<Func<AuditLog, bool>> whereExpression = p => p.ExecutionTime.Date >= startDate.Date && p.ExecutionTime.Date <= endDate.Date;

            if (searchString != string.Empty)
            {
                whereExpression = whereExpression.And(p => p.LoginName.Contains(searchString)
                                                        || p.UserName.Contains(searchString)
                                                        || p.ClientIpAddress.Contains(searchString)
                                                        || p.ClientName.Contains(searchString)
                                                        || p.MethodName.Contains(searchString));
            }
            if (!string.IsNullOrEmpty(hospitalNumber))
            {
                whereExpression = whereExpression.And(p => p.HospitalNumber == hospitalNumber);
            }
            Entity.Page<AuditLog> pageLogs = DBServices.ElincloudLogInstance.GetMuchByPage<AuditLog>(new DBPage(pageSize, pageIndex), whereExpression, query => query.OrderByDescending(x => x.ExecutionTime), true);
            foreach (var log in pageLogs.Content)
            {
                AuditLogInfo logInfo = new AuditLogInfo()
                {
                    LoginName = log.LoginName,
                    UserName = log.UserName,
                    ServiceName = log.ServiceName,
                    MethodName = log.MethodName,
                    Parameters = log.Parameters,
                    ReturnValue = log.ReturnValue,
                    ExecutionTime = log.ExecutionTime,
                    ExecutionDuration = log.ExecutionDuration,
                    ClientIpAddress = log.ClientIpAddress,
                    ClientName = log.ClientName,
                    BrowserInfo = log.BrowserInfo,
                    Exception = log.Exception,
                    ExceptionMessage = log.ExceptionMessage,
                    CustomData = log.CustomData,
                    HospitalNumber = log.HospitalNumber
                };
                logInfos.Add(logInfo);
            }
            return new Entity.Page<AuditLogInfo>(pageSize, pageIndex, pageLogs.TotalElements, logInfos);
        }

        #endregion

        #region 错误日志

        public List<ErrorLogInfo> GetErrorLogInfos(DateTime logDate, LogTypes logType, string hospitalNumber = "")
        {
            List<ErrorLogInfo> errorLogInfos = new List<ErrorLogInfo>();
            string logFolder = getLogFolder(logDate, logType, ApplicationEx.ErrorLogDirectoryName, hospitalNumber);
            if (System.IO.Directory.Exists(logFolder))
            {
                string[] logFiles = Directory.GetFiles(logFolder).OrderBy(f => Utils.StringUtil.GetFileName(f)).ToArray();
                foreach (var file in logFiles)
                {
                    FileInfo fileInfo = new FileInfo(file);
                    string fileName = Utils.StringUtil.GetFileNameWithoutExtension(fileInfo.Name); // 提取 "2025061711"
                    bool isDate = DateTime.TryParseExact(fileName, "yyyyMMddHH", System.Globalization.CultureInfo.InvariantCulture, System.Globalization.DateTimeStyles.None, out DateTime date);
                    if (isDate)
                    {
                        if (fileInfo.Extension.ToUpper() == ".TXT")
                        {
                            errorLogInfos.Add(new ErrorLogInfo()
                            {
                                LogFileName = fileInfo.Name,
                                LogFileSize = ElincloudServer.Common.Utils.FileUtil.FormatFileSize(fileInfo.Length),
                            }); ;
                        }
                    }

                }
            }
            errorLogInfos = errorLogInfos.OrderBy(e => e.LogFileName).ToList();
            return errorLogInfos;
        }

        public string GetErrorLogContent(DateTime logDate, LogTypes logType, string logFileName, string hospitalNumber = "")
        {
            string logFolder = getLogFolder(logDate, logType, ApplicationEx.ErrorLogDirectoryName, hospitalNumber);
            string logFileFullName = ElincloudServer.Common.Utils.StringUtil.Combine(logFolder, logFileName);
            if (System.IO.File.Exists(logFileFullName))
            {
                return System.IO.File.ReadAllText(logFileFullName);
            }
            else
            {
                throw new ServiceException("W00000315");
            }

        }

        #endregion

        #region 运行日志

        public List<ErrorLogInfo> GetRunLogInfos(DateTime logDate, LogTypes logType, string hospitalNumber = "")
        {
            List<ErrorLogInfo> errorLogInfos = new List<ErrorLogInfo>();
            string logFolder = getLogFolder(logDate, logType, ApplicationEx.RunLogDirectoryName, hospitalNumber);
            if (System.IO.Directory.Exists(logFolder))
            {
                string[] logFiles = Directory.GetFiles(logFolder).OrderBy(f => Utils.StringUtil.GetFileName(f)).ToArray();
                foreach (var file in logFiles)
                {
                    FileInfo fileInfo = new FileInfo(file);
                    string fileName = Utils.StringUtil.GetFileNameWithoutExtension(fileInfo.Name); // 提取 "2025061711"
                    bool isDate = DateTime.TryParseExact(fileName, "yyyyMMddHH", System.Globalization.CultureInfo.InvariantCulture, System.Globalization.DateTimeStyles.None, out DateTime date);
                    if (isDate)
                    {
                        if (fileInfo.Extension.ToUpper() == ".TXT")
                        {
                            errorLogInfos.Add(new ErrorLogInfo()
                            {
                                LogFileName = fileInfo.Name,
                                LogFileSize = ElincloudServer.Common.Utils.FileUtil.FormatFileSize(fileInfo.Length),
                            }); ;
                        }
                    }

                }
            }
            errorLogInfos = errorLogInfos.OrderBy(e => e.LogFileName).ToList();
            return errorLogInfos;
        }

        public string GetRunLogContent(DateTime logDate, LogTypes logType, string logFileName, string hospitalNumber = "")
        {
            string logFolder = getLogFolder(logDate, logType, ApplicationEx.RunLogDirectoryName, hospitalNumber);
            string logFileFullName = ElincloudServer.Common.Utils.StringUtil.Combine(logFolder, logFileName);
            if (System.IO.File.Exists(logFileFullName))
            {
                return System.IO.File.ReadAllText(logFileFullName);
            }
            else
            {
                throw new ServiceException("W00000315");
            }

        }

        #endregion

        #region 私有方法

        private string getLogFolder(DateTime logDate, LogTypes logType, string directoryName, string hospitalNumber = "")
        {
            string logFolder = "";
            string dateFolder = logDate.ToString("yyyyMMdd");
            if (logType == LogTypes.服务器)
            {
                logFolder = ElincloudServer.Common.Utils.StringUtil.Combine(AppDomain.CurrentDomain.BaseDirectory, ElincloudServer.Common.ApplicationEx.ServiceLogFolder, directoryName, dateFolder);
            }
            else
            {
                ServiceCommonConfigInfo configInfo = CommonBL.GetServiceCommonConfigInfo();
                if (logType == LogTypes.前置机)
                {
                    logFolder = ElincloudServer.Common.Utils.StringUtil.Combine(configInfo.FileStoragePath, ElincloudServer.Common.ApplicationEx.ForwardingSystemLogFolder, hospitalNumber, directoryName, dateFolder);
                }
            }

            return logFolder;
        }

        #endregion

        #endregion

        #region 地址管理

        /// <summary>
        /// 获取地址信息列表
        /// </summary>
        /// <param name="hospitalGroupID"></param>
        /// whz 2024年11月13日10:27:25
        /// <returns></returns>
        public List<AddressInfo> GetHospitalAddressInfos(string hospitalGroupID)
        {
            List<AddressInfo> addressInfos = new List<AddressInfo>();
            SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
            List<Address> addresses = instance.GetMuch<Address>();

            if (addresses != null && addresses.Count > 0)
            {
                foreach (Address each in addresses)
                {
                    addressInfos.Add(new AddressInfo
                    {
                        ID = each.ID,
                        PID = each.PID,
                        PostCode = each.PostCode,
                        Name = each.Name,
                        Code = each.Code,
                    });
                }
            }
            return addressInfos;
        }

        /// <summary>
        ///   添加地址信息
        /// </summary>
        /// <param name="pi"></param>
        /// <param name="hospitalGroupID"></param>
        /// <param name="addressInfo"></param>
        /// <exception cref="ServiceException"></exception>
        public void AddAddressInfo(PermissionInfo pi, string hospitalGroupID, AddressInfo addressInfo)
        {
            if (addressInfo != null)
            {
                SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
                int count = instance.GetCount<Address>(c => c.Name == addressInfo.Name && c.PID == addressInfo.PID);
                if (count > 0)
                {
                    throw new ServiceException(("W00000215"));
                }
                Address address = new Address()
                {
                    ID = instance.GetID<Address>(),
                    Name = addressInfo.Name,
                    Code = addressInfo.Code,
                    PostCode = addressInfo.PostCode,
                    PID = addressInfo.PID,
                    UpdateDate = DateTime.Now,
                    UpdateUserID = pi.UserID,
                    CreateDate = DateTime.Now,
                    CreateUserID = pi.UserID,
                };
                instance.Insert<Address>(address);
            }
            else
            {
                throw new ServiceException("W00000216");
            }
        }

        /// <summary>
        /// 更新地址信息
        /// </summary>
        /// <param name="pi"></param>
        /// <param name="hospitalGroupID"></param>
        /// <param name="addressInfo"></param>
        /// whz 2024年11月13日10:27:42
        /// <exception cref="ServiceException"></exception>
        public void UpdateAddressInfo(PermissionInfo pi, string hospitalGroupID, AddressInfo addressInfo)
        {
            if (addressInfo != null)
            {
                SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
                Address address = instance.GetOne<Address>(p => p.ID == addressInfo.ID);
                if (address != null)
                {
                    int count = instance.GetCount<Address>(c => c.Name == addressInfo.Name && c.PID == addressInfo.PID && c.ID != addressInfo.ID);
                    if (count > 0)
                    {
                        throw new ServiceException("W00000217");
                    }
                    address.Code = addressInfo.Code;
                    address.PostCode = addressInfo.PostCode;
                    address.Name = addressInfo.Name;
                    address.UpdateDate = DateTime.Now;
                    address.UpdateUserID = pi.UserID;
                    instance.Update<Address>(address, add => add.ID == address.ID, Address._Name, Address._Code, Address._PostCode, Address._UpdateDate, Address._UpdateUserID);
                }
                else
                {
                    throw new ServiceException("W00000219");
                }
            }
            else
            {
                throw new ServiceException("W00000218");
            }
        }
        /// <summary>
        /// 删除地址信息
        /// </summary>
        /// <param name="hospitalGroupID"></param>
        /// <param name="addressID"></param>
        /// <exception cref="ServiceException"></exception>
        public void DeleteAddressInfo(string hospitalGroupID, string addressID)
        {
            SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
            int count = instance.GetCount<Address>(p => p.PID == addressID);
            if (count == 0)
            {
                instance.Delete<Address>(add => add.ID == addressID);
            }
            else
            {
                throw new ServiceException("W00000220");
            }
        }

        /// <summary>
        /// 根据pid 查询地址
        /// </summary>
        /// <param name="hospitalGroupID"></param>
        /// <param name="parentID"></param>
        /// <returns></returns>
        public List<AddressInfo> GetAddressInfosByPID(string hospitalGroupID, string searchStr, string parentID)
        {
            List<AddressInfo> ret = new List<AddressInfo>();
            SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);

            Expression<Func<Address, bool>> whereExpression = x => x.PID == parentID;
            if (!string.IsNullOrEmpty(searchStr))
            {
                whereExpression = whereExpression.And(x => x.Name.Contains(searchStr));
            }
            List<Address> addresses = instance.GetMuch<Address>(whereExpression);

            if (addresses != null && addresses.Count > 0)
            {
                foreach (Address item in addresses)
                {
                    ret.Add(new AddressInfo
                    {
                        ID = item.ID,
                        PID = item.PID,
                        PostCode = item.PostCode,
                        Name = item.Name,
                        Code = item.Code,
                    });
                }
            }
            return ret;
        }

        #endregion



        #region 用户登录

        public AccountInfo Login(string loginName, string password, bool isAdmin, Action<int> checkAuthorizeCallBack = null)
        {
            //过滤登录账号中的空格
            string _loginName = loginName.Trim();
            if (isAdmin)//运维登录
            {
                //lamda方式取到的用户登录信息集合中存在,用户名中大写和小写两个账号 如 admin和Admin都会取到，原因是数据库不区分大小写
                List<Admin> accounts = DBServices.ElincloudBasicInstance.GetMuch<Admin>(p => p.LoginName == _loginName && p.Password == password);
                if (accounts != null && accounts.Count > 0)
                {
                    //此处用登录名获取一个用户 是可以区分登录名的大小写的
                    Admin account = accounts.FirstOrDefault(p => p.LoginName == _loginName);
                    if (account != null)
                    {
                        if (account.UserState == (int)SysStatus.启用)
                        {
                            List<string> hospitalIDs = new List<string>();
                            if (account.LoginName != Common.ApplicationEx.SuperAdmin)
                            {
                                List<AdminRoleRela> adminRoleRelas = DBServices.ElincloudBasicInstance.GetMuch<AdminRoleRela>(p => p.AdminID == account.ID);
                                if (adminRoleRelas != null && adminRoleRelas.Count > 0)
                                {
                                    List<string> roleIDs = adminRoleRelas.Select(p => p.RoleID).ToList();
                                    List<AdminRoleHospitalRela> adminRoleHospitalRelas = DBServices.ElincloudBasicInstance.GetMuch<AdminRoleHospitalRela>(p => roleIDs.Contains(p.RoleID));
                                    if (adminRoleHospitalRelas != null && adminRoleHospitalRelas.Count > 0)
                                    {
                                        hospitalIDs = adminRoleHospitalRelas.Select(p => p.HospitalID).ToList();
                                    }
                                }
                            }
                            else
                            {
                                List<Hospital> hospitals = DBServices.ElincloudBasicInstance.GetMuch<Hospital>();
                                hospitalIDs = hospitals.Select(p => p.ID).ToList();
                            }

                            AccountInfo accountInfo = new AccountInfo()
                            {
                                LoginInfo = new LoginInfo()
                                {
                                    UserID = account.ID,
                                    UserName = account.UserName,
                                    LoginName = account.LoginName,
                                    Password = account.Password,
                                    IsAdmin = isAdmin,
                                    HospitalIDs = hospitalIDs//运维账号可以维护的医院
                                },
                                User = new UserInfo()
                                {
                                    ID = account.ID,
                                    LoginName = _loginName,
                                    Password = account.Password,
                                    Email = account.Email,
                                    TelePhone = account.TelePhone,
                                    UserState = ElincloudServer.Common.Utils.EnumUtil.GetEnumValue<SysStatus>(account.UserState),
                                }
                            };
                            return accountInfo;
                        }
                        else
                        {
                            throw new ServiceException("W00000149");
                        }
                    }
                    else
                    {
                        throw new ServiceException("W00000150");
                    }
                }
                else
                {
                    throw new ServiceException("W00000150");
                }
            }
            else//医院用户登录
            {
                //lamda方式取到的用户登录信息集合中存在,用户名中大写和小写两个账号 如 admin和Admin都会取到，原因是数据库不区分大小写
                string where = DBServices.ElincloudBasicInstance.GetWhereByExpression<Doctor>(p => p.LoginName == loginName && p.Password == password);
                List<Doctor> docs = DBServices.ElincloudBasicInstance.GetMuchByProcedure<Doctor>(ElincloudServer.Common.ApplicationEx.MultiHospitalProcName, typeof(Doctor).Name, where);

                if (docs != null && docs.Count > 0)
                {
                    //此处用登录名获取一个用户 是可以区分登录名的大小写的
                    Doctor doctor = docs.FirstOrDefault(p => p.LoginName == _loginName);
                    if (doctor != null)
                    {
                        if (doctor.UserState == (int)SysStatus.启用)
                        {
                            HospitalRegist hospitalRegist = DBServices.ElincloudBasicInstance.GetOne<HospitalRegist>(c => c.HospitalID == doctor.HospitalID);
                            if (hospitalRegist != null)
                            {
                                int remaindays = ElincloudServer.Common.ApplicationEx.GetRemainDays(hospitalRegist.RegistCode, hospitalRegist.RegistKey);
                                if (remaindays < 0)
                                {
                                    throw new ServiceException("W00000151");
                                }
                                else if (hospitalRegist.HospitalStatus == (int)CloudImageHospitalStatus.停用)
                                {
                                    throw new ServiceException("W00000152");
                                }
                                else
                                {
                                    //通知Controller权限剩余天数，当小于30的时候会给用户提醒
                                    if (checkAuthorizeCallBack != null)
                                    {
                                        checkAuthorizeCallBack(remaindays);
                                    }

                                    AccountInfo accountInfo = new AccountInfo()
                                    {
                                        LoginInfo = new LoginInfo()
                                        {
                                            UserID = doctor.ID,
                                            UserName = doctor.Name,
                                            LoginName = doctor.LoginName,
                                            Password = doctor.Password,
                                            IsAdmin = false,
                                        },
                                        User = new UserInfo()
                                        {
                                            ID = doctor.ID,
                                            UserName = doctor.Name,
                                            LoginName = _loginName,
                                            Password = doctor.Password,
                                            Email = doctor.Email,
                                            Remark = doctor.Remark,
                                            Sex = ElincloudServer.Common.Utils.EnumUtil.GetEnumValue<Sexs>(doctor.Sex),
                                            TelePhone = doctor.TelePhone,
                                            Birthday = doctor.Birthday,
                                            UserState = ElincloudServer.Common.Utils.EnumUtil.GetEnumValue<SysStatus>(doctor.UserState),
                                            JobTitle = doctor.JobTitle,
                                            Address = doctor.Address,
                                            HeadPortraitRelaFileName = doctor.HeadPortraitRelaFileName,
                                            DiagnosisSignatureRelaFileName = doctor.DiagnosisSignatureRelaFileName,
                                        }
                                    };

                                    SqlAccess instance = DBServices.GetHospitalDBInstance(doctor.HospitalGroupID);

                                    //获取角色信息
                                    accountInfo.UserRoles = CommonBL.GetUserRolesByUserID(instance, doctor.ID);
                                    if (accountInfo.UserRoles == null || accountInfo.UserRoles.Count == 0)
                                    {
                                        throw new ServiceException("W00000153");
                                    }
                                    //组织机构信息
                                    accountInfo.UserOrganization = CommonBL.GetUserOrganizationByUserID(instance, doctor.ID);
                                    if (accountInfo.UserOrganization == null)
                                    {
                                        throw new ServiceException("W00000154");
                                    }
                                    //医院信息
                                    accountInfo.UserHospital = CommonBL.GetHospitalInfoByID(doctor.HospitalID);
                                    if (accountInfo.UserHospital == null)
                                    {
                                        throw new ServiceException("W00000155");
                                    }

                                    return accountInfo;

                                }
                            }
                            else
                            {
                                throw new ServiceException("W00000156");
                            }
                        }
                        else
                        {
                            throw new ServiceException("W00000149");
                        }
                    }
                    else
                    {
                        throw new ServiceException("W00000150");
                    }
                }
                else
                {
                    throw new ServiceException("W00000150");
                }
            }
        }

        /// <summary> 
        /// 更新密码
        /// </summary>
        /// <param name="pi"></param>
        /// <param name="newPwd"></param>
        /// <param name="user"></param>
        public void UpdatePassword(PermissionInfo pi, string userID, string oldPwd, string newPwd)
        {
            if (pi.IsAdmin)
            {
                Admin account = DBServices.ElincloudBasicInstance.GetOne<Admin>(p => p.ID == userID);
                if (account != null)
                {
                    if (account.Password == oldPwd)
                    {
                        account.Password = newPwd;
                        account.UpdateDate = DateTime.Now;
                        account.UpdateUserID = pi.UserID;
                        //更新密码
                        DBServices.ElincloudBasicInstance.Update<Admin>(account, p => p.ID == userID, Admin._Password, Admin._UpdateDate, Admin._UpdateUserID);
                    }
                    else
                    {
                        throw new ServiceException("W00000158");
                    }
                }
                else
                {
                    throw new ServiceException("W00000159");
                }
            }
            else
            {
                string where = DBServices.ElincloudBasicInstance.GetWhereByExpression<Doctor>(p => p.ID == userID);
                List<Doctor> docs = DBServices.ElincloudBasicInstance.GetMuchByProcedure<Doctor>(ElincloudServer.Common.ApplicationEx.MultiHospitalProcName, typeof(Doctor).Name, where);
                Doctor doctor = docs.FirstOrDefault();
                if (doctor != null)
                {
                    if (doctor.Password == oldPwd)
                    {
                        CommonBL.PasswordPolicyVerification(pi.HospitalCode, newPwd);

                        doctor.Password = newPwd;
                        doctor.UpdateDate = DateTime.Now;
                        doctor.UpdateUserID = pi.UserID;
                        //更新密码
                        SqlAccess instance = DBServices.GetHospitalDBInstance(doctor.HospitalGroupID);
                        instance.Update<Doctor>(doctor, p => p.ID == userID, Admin._Password, Admin._UpdateDate, Admin._UpdateUserID);
                    }
                    else
                    {
                        throw new ServiceException("W00000158");
                    }
                }
                else
                {
                    throw new ServiceException("W00000159");
                }
            }
        }

        #endregion

        #region 用户管理
        public UserInfo GetUserByID(string hospitalGroupID, string userID)
        {
            SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
            Doctor doc = instance.GetOne<Doctor>(p => p.ID == userID);
            if (doc != null)
            {
                UserInfo user = new UserInfo()
                {
                    ID = doc.ID,
                    UserName = doc.Name,
                    LoginName = doc.LoginName,
                    Password = doc.Password,
                    UserState = ElincloudServer.Common.Utils.EnumUtil.GetEnumValue<SysStatus>(doc.UserState),
                    TelePhone = doc.TelePhone,
                    CertificateCode = doc.CertificateCode,
                    Email = doc.Email,
                    JobTitle = doc.JobTitle,
                    Address = doc.Address,
                    Birthday = doc.Birthday,
                    HeadPortraitRelaFileName = doc.HeadPortraitRelaFileName,
                    Remark = doc.Remark,
                    Sex = ElincloudServer.Common.Utils.EnumUtil.GetEnumValue<Sexs>(doc.Sex),
                    DiagnosisSignatureRelaFileName = doc.DiagnosisSignatureRelaFileName,
                    IsAdmin = false,
                };
                return user;
            }
            else
            {
                throw new ServiceException("W00000170");
            }
        }

        /// <summary>
        /// 更新用户名称获取用户信息
        /// </summary>
        /// <param name="hospitalGroupID"></param>
        /// <param name="userName"></param>
        /// <returns></returns>
        /// <exception cref="ServiceException"></exception>
        public UserInfo GetUserByName(string hospitalGroupID, string userName)
        {
            SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
            Doctor doc = instance.GetOne<Doctor>(p => p.Name == userName);
            if (doc != null)
            {
                UserInfo user = new UserInfo()
                {
                    ID = doc.ID,
                    UserName = doc.Name,
                    LoginName = doc.LoginName,
                    TelePhone = doc.TelePhone,
                    CertificateCode = doc.CertificateCode,
                    Email = doc.Email,
                    JobTitle = doc.JobTitle,
                    Sex = ElincloudServer.Common.Utils.EnumUtil.GetEnumValue<Sexs>(doc.Sex),
                    IsAdmin = false,
                };
                return user;
            }
            else
            {
                throw new ServiceException("W00000170");
            }
        }

        public DetailUserInfo GetDetailUserInfoByID(string hospitalGroupID, string userID)
        {
            DetailUserInfo detail = new DetailUserInfo();
            detail.User = GetUserByID(hospitalGroupID, userID);
            SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
            detail.UserOrganization = CommonBL.GetUserOrganizationByUserID(instance, userID);
            detail.UserRoles = CommonBL.GetUserRolesByUserID(instance, userID);
            return detail;
        }

        /// <summary>
        /// 查询用户列表数据
        /// </summary>
        /// <param name="pi"></param>
        /// <param name="priority"></param>
        /// <returns></returns>
        public List<UserInfo> GetHospitalUsers(string hospitalID, string hospitalGroupID)
        {
            List<UserInfo> userInfos = new List<UserInfo>();
            SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
            List<Doctor> docs = instance.GetMuch<Doctor>(p => p.HospitalID == hospitalID);
            foreach (var doc in docs)
            {
                UserInfo user = new UserInfo()
                {
                    ID = doc.ID,
                    UserName = doc.Name,
                    LoginName = doc.LoginName,
                    Password = doc.Password,
                    Birthday = doc.Birthday,
                    UserState = ElincloudServer.Common.Utils.EnumUtil.GetEnumValue<SysStatus>(doc.UserState),
                    TelePhone = doc.TelePhone,
                    CertificateCode = doc.CertificateCode,
                    Email = doc.Email,
                };
                userInfos.Add(user);
            }
            return userInfos;
        }

        /// <summary>
        /// 查询用户列表数据
        /// </summary>
        /// <param name="hospitalID"></param>
        /// <param name="searchStr"></param>
        /// <param name="pageIndex"></param>
        /// <param name="pageSize"></param>
        /// <returns></returns>
        public Entity.Page<DoctorInfo> GetHospitalUsersByPage(string hospitalID, string hospitalGroupID, string searchStr, int pageIndex, int pageSize)
        {
            List<DoctorInfo> userInfos = new List<DoctorInfo>();
            SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
            Expression<Func<Doctor, bool>> whereExpression = p => p.HospitalID == hospitalID;
            if (!string.IsNullOrEmpty(searchStr))
            {
                whereExpression = whereExpression.And(p => (p.LoginName.Contains(searchStr) || p.TelePhone.Contains(searchStr) || p.Name.Contains(searchStr)));
            }
            Entity.Page<Doctor> pageDoctor = instance.GetMuchByPage<Doctor>(new DBPage(pageSize, pageIndex), whereExpression);
            List<DoctorRoleRela> doctorRoles = instance.GetMuch<DoctorRoleRela>();
            List<DoctorOrganizationRela> doctorOrgs = instance.GetMuch<DoctorOrganizationRela>();
            foreach (var doc in pageDoctor.Content)
            {
                DoctorInfo user = new DoctorInfo()
                {
                    ID = doc.ID,
                    UserName = doc.Name,
                    LoginName = doc.LoginName,
                    Password = doc.Password,
                    Birthday = doc.Birthday,
                    UserState = ElincloudServer.Common.Utils.EnumUtil.GetEnumValue<SysStatus>(doc.UserState),
                    TelePhone = doc.TelePhone,
                    Email = doc.Email,
                    Address = doc.Address,
                    Sex = ElincloudServer.Common.Utils.EnumUtil.GetEnumValue<Sexs>(doc.Sex),
                    Remark = doc.Remark,
                    IsExistsOrg = doctorOrgs.Exists(p => p.DoctorID == doc.ID),
                    IsExistsRole = doctorRoles.Exists(p => p.DoctorID == doc.ID)
                };
                userInfos.Add(user);
            }
            return new Entity.Page<DoctorInfo>(pageSize, pageIndex, pageDoctor.TotalElements, userInfos);
        }

        /// <summary>
        /// 获取该组织机构所有用户列表
        /// </summary>
        /// <returns>用户列表</returns>
        public List<UserInfo> GetUsersByOrgnization(string hospitalGroupID, string orgID)
        {
            List<UserInfo> rets = new List<UserInfo>();
            SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
            List<DoctorOrganizationRela> doctorOrganizationRelas = instance.GetMuch<DoctorOrganizationRela>(p => p.OrganizationID == orgID);
            if (doctorOrganizationRelas != null)
            {
                List<string> doctorIds = doctorOrganizationRelas.Select(c => c.DoctorID).ToList();
                List<Doctor> doctors = instance.GetMuch<Doctor>(p => doctorIds.Contains(p.ID) && p.UserState == (int)SysStatus.启用);
                foreach (var item in doctors)
                {
                    UserInfo ret = new UserInfo()
                    {
                        ID = item.ID,
                        UserName = item.Name,
                        Email = item.Email,
                        TelePhone = item.TelePhone,
                        CertificateCode = item.CertificateCode,
                        Birthday = item.Birthday,
                        UserState = ElincloudServer.Common.Utils.EnumUtil.GetEnumValue<SysStatus>(item.UserState),
                        Address = item.Address,
                        Sex = ElincloudServer.Common.Utils.EnumUtil.GetEnumValue<Sexs>(item.Sex),
                        Remark = item.Remark
                    };
                    rets.Add(ret);
                }
            }
            return rets;
        }

        /// <summary>
        /// 获取该组织机构所有用户列表 分页
        /// </summary>
        /// <param name="hospitalID"></param>
        /// <param name="orgID"></param>
        /// <returns></returns>
        public Page<UserInfo> GetUsersByOrgnizationByPage(string hospitalGroupID, string searchStr, string orgID, int pageIndex, int pageSize)
        {
            List<UserInfo> rets = new List<UserInfo>();
            SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
            List<DoctorOrganizationRela> doctorOrganizationRelas = instance.GetMuch<DoctorOrganizationRela>(p => p.OrganizationID == orgID);
            if (doctorOrganizationRelas != null)
            {
                List<string> doctorIds = doctorOrganizationRelas.Select(c => c.DoctorID).ToList();

                #region 搜索
                Expression<Func<Doctor, bool>> whereExpression = p => doctorIds.Contains(p.ID);
                if (!string.IsNullOrEmpty(searchStr))
                {
                    whereExpression = whereExpression.And(x => x.Name.Contains(searchStr));
                }
                #endregion

                Page<Doctor> doctors = instance.GetMuchByPage<Doctor>(new DBPage(pageSize, pageIndex), whereExpression);
                foreach (var item in doctors.Content)
                {
                    UserInfo ret = new UserInfo()
                    {
                        ID = item.ID,
                        UserName = item.Name,
                        Email = item.Email,
                        TelePhone = item.TelePhone,
                        CertificateCode = item.CertificateCode,
                        Birthday = item.Birthday,
                        UserState = ElincloudServer.Common.Utils.EnumUtil.GetEnumValue<SysStatus>(item.UserState),
                        Address = item.Address,
                        Sex = ElincloudServer.Common.Utils.EnumUtil.GetEnumValue<Sexs>(item.Sex),
                        Remark = item.Remark
                    };
                    rets.Add(ret);
                }
                return new Entity.Page<UserInfo>(pageSize, pageIndex, doctors.TotalElements, rets);
            }
            return new Entity.Page<UserInfo>(pageSize, pageIndex, 0, rets);

        }

        /// <summary>
        /// 根据角色ID获取用户列表
        /// </summary>
        /// <param name="roleID"></param>
        /// <returns></returns>
        public List<UserInfo> GetUsersByRole(string hospitalGroupID, string roleID)
        {
            List<UserInfo> rets = new List<UserInfo>();
            SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
            List<DoctorRoleRela> doctorRoleRelas = instance.GetMuch<DoctorRoleRela>(p => p.RoleID == roleID);
            if (doctorRoleRelas != null)
            {
                List<string> doctorIDs = doctorRoleRelas.Select(c => c.DoctorID).ToList();
                if (doctorIDs != null)
                {
                    List<Doctor> doctors = instance.GetMuch<Doctor>(p => doctorIDs.Contains(p.ID) && p.UserState == (int)SysStatus.启用);
                    if (doctors != null)
                    {
                        foreach (var item in doctors)
                        {
                            UserInfo ret = new UserInfo()
                            {
                                ID = item.ID,
                                UserName = item.Name,
                                LoginName = item.LoginName,
                                Email = item.Email,
                                TelePhone = item.TelePhone,
                                Birthday = item.Birthday,
                                UserState = ElincloudServer.Common.Utils.EnumUtil.GetEnumValue<SysStatus>(item.UserState),
                                Address = item.Address,
                                Sex = ElincloudServer.Common.Utils.EnumUtil.GetEnumValue<Sexs>(item.Sex),
                                Remark = item.Remark,
                                CertificateCode = item.CertificateCode
                            };
                            rets.Add(ret);
                        }
                    }
                }
            }
            return rets;
        }

        /// <summary>
        ///  根据角色ID获取用户列表 分页
        /// </summary>
        /// <param name="hospitalID"></param>
        /// <param name="roleID"></param>
        /// <param name="pageIndex"></param>
        /// <param name="pageSize"></param>
        /// <returns></returns>
        public Page<UserInfo> GetUsersByRoleByPage(string hospitalGroupID, string roleID, int pageIndex, int pageSize)
        {
            List<UserInfo> rets = new List<UserInfo>();
            SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
            List<DoctorRoleRela> doctorRoleRelas = instance.GetMuch<DoctorRoleRela>(p => p.RoleID == roleID);
            if (doctorRoleRelas != null)
            {
                List<string> doctorIDs = doctorRoleRelas.Select(c => c.DoctorID).ToList();
                if (doctorIDs != null)
                {
                    Page<Doctor> doctors = instance.GetMuchByPage<Doctor>(new DBPage(pageSize, pageIndex), p => doctorIDs.Contains(p.ID));
                    if (doctors.Content != null)
                    {
                        foreach (var item in doctors.Content)
                        {
                            UserInfo ret = new UserInfo()
                            {
                                ID = item.ID,
                                UserName = item.Name,
                                Email = item.Email,
                                TelePhone = item.TelePhone,
                                Birthday = item.Birthday,
                                UserState = ElincloudServer.Common.Utils.EnumUtil.GetEnumValue<SysStatus>(item.UserState),
                                Address = item.Address,
                                Sex = ElincloudServer.Common.Utils.EnumUtil.GetEnumValue<Sexs>(item.Sex),
                                Remark = item.Remark,
                                CertificateCode = item.CertificateCode
                            };
                            rets.Add(ret);
                        }
                        return new Entity.Page<UserInfo>(pageSize, pageIndex, doctors.TotalElements, rets); ;
                    }
                }
            }
            return new Entity.Page<UserInfo>(pageSize, pageIndex, 0, rets); ;

        }

        /// <summary>
        /// 设置用户启用、停用状态
        /// </summary>
        /// <param name="pi"></param>
        /// <param name="userID"></param>
        /// <param name="state"></param>
        /// <exception cref="ServiceException"></exception>
        public void UpdateUserStatus(PermissionInfo pi, string userID, SysStatus state)
        {
            string where = DBServices.ElincloudBasicInstance.GetWhereByExpression<Doctor>(p => p.ID == userID);
            List<Doctor> docs = DBServices.ElincloudBasicInstance.GetMuchByProcedure<Doctor>(ElincloudServer.Common.ApplicationEx.MultiHospitalProcName, typeof(Doctor).Name, where);
            Doctor doctor = docs.FirstOrDefault();
            if (doctor != null)
            {
                SqlAccess instance = DBServices.GetHospitalDBInstance(doctor.HospitalGroupID);
                doctor.UserState = (int)state;
                doctor.UpdateDate = DateTime.Now;
                doctor.UpdateUserID = pi.UserID;
                instance.Update<Doctor>(doctor, p => p.ID == userID, Admin._UserState, Admin._UpdateDate, Admin._UpdateUserID);
            }
            else
            {
                throw new ServiceException("W00000158");
            }
        }

        /// <summary>
        /// 新增用户
        /// </summary>
        /// <param name="user"></param>
        /// <returns>ID</returns>
        public void AddUser(PermissionInfo pi, string hospitalID, string hospitalGroupID, string hospitalGroupCode, UserInfo user)
        {
            if (user != null)
            {
                if (!ElincloudServer.Common.Utils.StringUtil.LoginNameValid(user.LoginName))
                {
                    throw new ServiceException("W00000172");
                }
                //添加用户时需要验证所有已存在用户的登录名是否存在
                List<Admin> accounts = DBServices.ElincloudBasicInstance.GetMuch<Admin>(p => p.LoginName == user.LoginName);
                string where = DBServices.ElincloudBasicInstance.GetWhereByExpression<Doctor>(p => p.LoginName == user.LoginName || p.Name == user.UserName);
                List<Doctor> doctors = DBServices.ElincloudBasicInstance.GetMuchByProcedure<Doctor>(ElincloudServer.Common.ApplicationEx.MultiHospitalProcName, typeof(Doctor).Name, where);
                accounts.ForEach(p => doctors.Add(new Doctor() { ID = p.ID, LoginName = p.LoginName }));

                SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
                //这里再次加loginname相等的判断是为了解决登录名要区分大小写的问题
                Doctor doctor = doctors.FirstOrDefault(p => p.LoginName == user.LoginName);
                Doctor userdoctor = doctors.FirstOrDefault(p => p.Name == user.UserName);
                if (userdoctor != null)
                {
                    throw new ServiceException("W00000173");
                }
                #region 验证手机号

                if (!string.IsNullOrWhiteSpace(user.TelePhone))
                {
                    if (doctors.Exists(p => p.TelePhone == user.TelePhone))
                    {
                        throw new ServiceException("W00000454");
                    }
                }

                #endregion

                #region 验证身份证号

                if (!string.IsNullOrWhiteSpace(user.CertificateCode))
                {
                    if (doctors.Exists(p => p.CertificateCode == user.CertificateCode))
                    {
                        throw new ServiceException("W00000455");
                    }
                }

                #endregion

                CommonBL.PasswordPolicyVerification(hospitalGroupCode, user.Password);

                if (doctor == null)
                {
                    #region 医院库doctor表中添加用户信息

                    doctor = new Doctor()
                    {
                        ID = instance.GetID<Doctor>(6, hospitalGroupID),
                        Name = user.UserName,
                        LoginName = user.LoginName,
                        Password = user.Password,
                        TelePhone = user.TelePhone,
                        Sex = (int)user.Sex,
                        Remark = user.Remark,
                        Birthday = user.Birthday,
                        Email = user.Email,
                        HeadPortraitRelaFileName = user.HeadPortraitRelaFileName,
                        DiagnosisSignatureRelaFileName = user.DiagnosisSignatureRelaFileName,
                        JobTitle = user.JobTitle,
                        CertificateCode = user.CertificateCode,
                        Address = user.Address,
                        UserState = (int)user.UserState,
                        HospitalID = hospitalID,
                        CreateUserID = pi.UserID,
                        CreateDate = DateTime.Now,
                        UpdateUserID = pi.UserID,
                        UpdateDate = DateTime.Now,
                    };
                    instance.Insert<Doctor>(doctor);

                    #endregion
                }
                else
                {
                    throw new ServiceException("W00000174");
                }
            }
            else
            {
                throw new ServiceException("W00000175");
            }
        }

        /// <summary>
        /// 修改用户的角色
        /// </summary>
        /// <param name="pi"></param>
        /// <param name="userID">要添加的用户ID</param>
        /// <param name="roleIDs">角色列表</param>
        /// <returns></returns>
        public void UpdateUserRoleRelas(string hospitalGroupID, string userID, List<string> roleIDs)
        {
            if (roleIDs.Count > 0)
            {
                SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
                Doctor doctor = instance.GetOne<Doctor>(p => p.ID == userID);
                if (doctor != null)
                {
                    Transaction tran = instance.GetTransaction();
                    List<DoctorRoleRela> userRoleRelas = new List<DoctorRoleRela>();
                    foreach (var roleID in roleIDs)
                    {
                        userRoleRelas.Add(new DoctorRoleRela()
                        {
                            DoctorID = doctor.ID,
                            RoleID = roleID,
                        });
                    }
                    tran.Delete<DoctorRoleRela>(c => c.DoctorID == doctor.ID);
                    tran.Insert<DoctorRoleRela>(userRoleRelas.ToArray());
                    if (!tran.Commit())
                    {
                        throw new ServiceException("W00000176");
                    }
                }
                else
                {
                    throw new ServiceException("W00000177");
                }
            }
            else
            {
                throw new ServiceException("W00000168");
            }
        }

        /// <summary>
        /// 修改用户的组织机构
        /// </summary>
        /// <param name="pi"></param>
        /// <param name="userID">用户iD</param>
        /// <param name="orgID">机构ID</param>
        /// <returns></returns>
        public void UpdateUserOrganizationRela(string hospitalGroupID, string userID, string orgID)
        {
            if (!string.IsNullOrEmpty(orgID))
            {
                SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
                Doctor doctor = instance.GetOne<Doctor>(p => p.ID == userID);
                if (doctor != null)
                {
                    Transaction tran = instance.GetTransaction();
                    tran.Delete<DoctorOrganizationRela>(c => c.DoctorID == doctor.ID);
                    DoctorOrganizationRela userOrgRela = new DoctorOrganizationRela()
                    {
                        DoctorID = doctor.ID,
                        OrganizationID = orgID,
                    };
                    tran.Insert<DoctorOrganizationRela>(userOrgRela);
                    if (!tran.Commit())
                    {
                        throw new ServiceException("W00000179");
                    }
                }
                else
                {
                    throw new ServiceException("W00000177");
                }
            }
            else
            {
                throw new ServiceException("W00000181");
            }
        }

        /// <summary>
        /// 更新用户信息
        /// </summary>
        /// <param name="user"></param>
        public void UpdateUser(PermissionInfo pi, string hospitalGroupID, string hospitalGroupCode, UserInfo user)
        {
            if (user != null)
            {
                SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
                List<Doctor> users = instance.GetMuch<Doctor>();
                Doctor doc = users.FirstOrDefault(i => i.ID == user.ID);
                if (doc != null)
                {
                    if (users.FindAll(i => i.Name == user.UserName && i.ID != user.ID).Count > 0)
                    {
                        throw new ServiceException("W00000182");
                    }

                    #region 验证登录名


                    if (user.LoginName != doc.LoginName)
                    {
                        if (!ElincloudServer.Common.Utils.StringUtil.LoginNameValid(user.LoginName))
                        {
                            throw new ServiceException("W00000172");
                        }
                        List<Admin> accounts = DBServices.ElincloudBasicInstance.GetMuch<Admin>(p => p.LoginName == user.LoginName);
                        string where = DBServices.ElincloudBasicInstance.GetWhereByExpression<Doctor>(p => p.LoginName == user.LoginName);
                        List<Doctor> doctors = DBServices.ElincloudBasicInstance.GetMuchByProcedure<Doctor>(ElincloudServer.Common.ApplicationEx.MultiHospitalProcName, typeof(Doctor).Name, where);
                        accounts.ForEach(p => doctors.Add(new Doctor() { ID = p.ID, LoginName = p.LoginName }));

                        if (doctors.Count > 0)
                        {
                            throw new ServiceException("W00000174");
                        }
                    }

                    #endregion

                    #region 验证手机号

                    if (!string.IsNullOrWhiteSpace(user.TelePhone))
                    {
                        if (users.Exists(p => p.TelePhone == user.TelePhone && p.ID != user.ID))
                        {
                            throw new ServiceException("W00000454");
                        }
                    }

                    #endregion

                    #region 验证身份证号

                    if (!string.IsNullOrWhiteSpace(user.CertificateCode))
                    {
                        if (users.Exists(p => p.CertificateCode == user.CertificateCode && p.ID != user.ID))
                        {
                            throw new ServiceException("W00000455");
                        }
                    }

                    #endregion

                    // 验证
                    CommonBL.PasswordPolicyVerification(hospitalGroupCode, user.Password);

                    doc.Name = user.UserName;
                    doc.LoginName = user.LoginName;
                    doc.Sex = (int)user.Sex;
                    doc.Birthday = user.Birthday;
                    doc.Address = user.Address;
                    doc.TelePhone = user.TelePhone;
                    doc.Password = user.Password;
                    doc.Email = user.Email;
                    doc.Remark = user.Remark;
                    doc.HeadPortraitRelaFileName = user.HeadPortraitRelaFileName;
                    doc.DiagnosisSignatureRelaFileName = user.DiagnosisSignatureRelaFileName;
                    doc.CertificateCode = user.CertificateCode;
                    doc.JobTitle = user.JobTitle;
                    doc.UpdateUserID = pi.UserID;
                    doc.UpdateDate = DateTime.Now;

                    //更新用户信息
                    instance.Update<Doctor>(doc, o => o.ID == user.ID,
                        Doctor._Name,
                        Doctor._LoginName,
                        Doctor._Sex,
                        Doctor._Birthday,
                        Doctor._Address,
                        Doctor._TelePhone,
                        Doctor._Password,
                        Doctor._Email,
                        Doctor._Remark,
                        Doctor._HeadPortraitRelaFileName,
                        Doctor._DiagnosisSignatureRelaFileName,
                        Doctor._VerifierSignatureRelaFileName,
                        Doctor._CertificateCode,
                        Doctor._JobTitle,
                        Doctor._UpdateDate,
                        Doctor._UpdateUserID);
                }
                else
                {
                    throw new ServiceException("W00000184");
                }
            }
            else
            {
                throw new ServiceException("W00000175");
            }
        }

        /// <summary>
        /// 删除用户角色
        /// </summary>
        /// <param name="userID"></param>
        /// <param name="roleID"></param>
        /// <returns></returns>
        public void DeleteUserRole(string hospitalGroupID, string userID, string roleID)
        {
            if (!string.IsNullOrEmpty(roleID))
            {
                if (!string.IsNullOrEmpty(userID))
                {
                    SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
                    Doctor doctor = instance.GetOne<Doctor>(p => p.ID == userID);
                    if (doctor != null)
                    {
                        instance.Delete<DoctorRoleRela>(c => c.DoctorID == userID && c.RoleID == roleID);
                    }
                    else
                    {
                        throw new ServiceException("W00000177");
                    }
                }
                else
                {
                    throw new ServiceException("W00000185");
                }
            }
            else
            {
                throw new ServiceException("W00000168");
            }

        }

        /// <summary>
        /// 查询没有机构的人员列表
        /// </summary>
        public List<UserInfo> GetUsersWithoutOrganization(string hospitalID, string hospitalGroupID)
        {
            List<UserInfo> users = new List<UserInfo>();
            SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
            List<Doctor> doctors = instance.GetMuch<Doctor>(p => p.HospitalID == hospitalID);
            if (doctors.Count > 0)
            {
                List<DoctorOrganizationRela> orrList = instance.GetMuch<DoctorOrganizationRela>();
                if (orrList != null)
                {
                    List<string> userids = orrList.Select(c => c.DoctorID).ToList();
                    var subDoctors = doctors.FindAll(p => !userids.Contains(p.ID));
                    subDoctors.ForEach(p =>
                    {
                        UserInfo user = new UserInfo()
                        {
                            ID = p.ID,
                            UserName = p.Name,
                        };
                        users.Add(user);
                    });
                }
            }
            return users;
        }

        /// <summary>
        /// 更新用户电子签名信息
        /// </summary>
        /// <param name="pi"></param>
        /// <param name="userID"></param>
        /// <param name="signatureRelaFileName"></param>
        public void UpdateUserSignature(PermissionInfo pi, string hospitalGroupID, string userID, string signatureRelaFileName)
        {
            SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
            Doctor doctor = instance.GetOne<Doctor>(p => p.ID == userID);
            if (doctor != null)
            {
                doctor.DiagnosisSignatureRelaFileName = signatureRelaFileName;
                doctor.UpdateDate = DateTime.Now;
                doctor.UpdateUserID = pi.UserID;
                instance.Update<Doctor>(doctor, p => p.ID == userID, Doctor._DiagnosisSignatureRelaFileName, Doctor._UpdateDate, Doctor._UpdateUserID);
            }
            else
            {
                throw new ServiceException("W00000186");
            }
        }

        /// <summary>
        /// 获取简单用户信息
        /// </summary>
        /// <param name="hospitalID"></param>
        /// <param name="pageIndex"></param>
        /// <param name="pageSize"></param>
        /// <returns></returns>
        public Page<SimpleUserInfo> GetSimpleUserInfosByPage(string hospitalID, string hospitalGroupID, string searchStr, int pageIndex, int pageSize)
        {
            List<SimpleUserInfo> rets = new List<SimpleUserInfo>();
            SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
            Expression<Func<Doctor, bool>> whereExpression = p => p.HospitalID == hospitalID;
            if (!string.IsNullOrEmpty(searchStr))
            {
                whereExpression = whereExpression.And(p => p.Name.Contains(searchStr));
            }
            Page<Doctor> doctors = instance.GetMuchByPage<Doctor>(new DBPage(pageSize, pageIndex), whereExpression);
            if (doctors.Content != null)
            {
                foreach (var item in doctors.Content)
                {
                    SimpleUserInfo ret = new SimpleUserInfo()
                    {
                        ID = item.ID,
                        UserName = item.Name
                    };
                    rets.Add(ret);
                }
            }
            return new Entity.Page<SimpleUserInfo>(pageSize, pageIndex, doctors.TotalElements, rets);
        }

        /// <summary>
        /// 获取简单用户信息
        /// </summary>
        /// <param name="hospitalID"></param>
        /// <param name="pageIndex"></param>
        /// <param name="pageSize"></param>
        /// <returns></returns>
        public Page<SimpleUserInfo> GetSimpleUserInfosByRoleByPage(string hospitalGroupID, string roleID, string searchStr, int pageIndex, int pageSize)
        {
            List<SimpleUserInfo> rets = new List<SimpleUserInfo>();
            SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);

            List<DoctorRoleRela> doctorRoleRelas = instance.GetMuch<DoctorRoleRela>(p => p.RoleID == roleID);
            if (doctorRoleRelas != null)
            {
                List<string> doctorIDs = doctorRoleRelas.Select(c => c.DoctorID).ToList();
                if (doctorIDs != null)
                {
                    Expression<Func<Doctor, bool>> whereExpression = p => doctorIDs.Contains(p.ID);
                    if (!string.IsNullOrEmpty(searchStr))
                    {
                        whereExpression = whereExpression.And(p => p.Name.Contains(searchStr));
                    }
                    Page<Doctor> doctors = instance.GetMuchByPage<Doctor>(new DBPage(pageSize, pageIndex), whereExpression);
                    if (doctors.Content != null)
                    {
                        foreach (var item in doctors.Content)
                        {
                            SimpleUserInfo ret = new SimpleUserInfo()
                            {
                                ID = item.ID,
                                UserName = item.Name
                            };
                            rets.Add(ret);
                        }
                        return new Entity.Page<SimpleUserInfo>(pageSize, pageIndex, doctors.TotalElements, rets);
                    }
                }
            }
            return new Entity.Page<SimpleUserInfo>(pageSize, pageIndex, 0, rets);

        }

        /// <summary>
        /// 获取简单用户信息
        /// </summary>
        /// <param name="hospitalID"></param>
        /// <returns></returns>
        public List<SimpleUserInfo> GetSimpleUserInfos(string hospitalID, string hospitalGroupID)
        {
            List<SimpleUserInfo> rets = new List<SimpleUserInfo>();
            SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
            List<Doctor> doctors = instance.GetMuch<Doctor>(p => p.HospitalID == hospitalID);
            if (doctors != null)
            {
                foreach (var item in doctors)
                {
                    SimpleUserInfo ret = new SimpleUserInfo()
                    {
                        ID = item.ID,
                        UserName = item.Name
                    };
                    rets.Add(ret);
                }
            }
            return rets;
        }

        #region 注册相关

        public void SendVerificationCode(string hospitalID, string telePhone)
        {
            if (!string.IsNullOrWhiteSpace(telePhone))
            {
                if (!string.IsNullOrWhiteSpace(hospitalID))
                {
                    Hospital hospital = DBServices.ElincloudBasicInstance.GetOne<Hospital>(i => i.ID == hospitalID);
                    HospitalRegist hospitalRegist = DBServices.ElincloudBasicInstance.GetOne<HospitalRegist>(c => c.HospitalID == hospitalID);
                    if (hospitalRegist != null && hospital != null)
                    {
                        int remaindays = ElincloudServer.Common.ApplicationEx.GetRemainDays(hospitalRegist.RegistCode, hospitalRegist.RegistKey);
                        if (remaindays > 0 && hospitalRegist.HospitalStatus == (int)CloudImageHospitalStatus.启用)
                        {
                            ElincloudServer.Common.Utils.SMSUtil.SendVerificationCodeToTelePhone(telePhone, hospital.Name, SMSUtil.UserRegisterStr);
                        }
                        else
                        {
                            throw new ServiceException("W00000365");
                        }
                    }
                    else
                    {
                        throw new ServiceException("W00000367");
                    }
                }
                else
                {
                    throw new ServiceException("W00000368");
                }
            }
            else
            {
                throw new ServiceException("W00000369");
            }
        }

        public void RegisterUserInfo(string hospitalID, string hospitalGroupID, UserInfo user, string roleID, string orgID, string operationUserID, string registerURL)
        {
            if (user != null)
            {
                if (!ElincloudServer.Common.Utils.StringUtil.LoginNameValid(user.LoginName))
                {
                    throw new ServiceException("W00000172");
                }
                //添加用户时需要验证所有已存在用户的登录名是否存在
                List<Admin> accounts = DBServices.ElincloudBasicInstance.GetMuch<Admin>(p => p.LoginName == user.LoginName);
                string where = DBServices.ElincloudBasicInstance.GetWhereByExpression<Doctor>(p => p.LoginName == user.LoginName || p.Name == user.UserName);
                List<Doctor> doctors = DBServices.ElincloudBasicInstance.GetMuchByProcedure<Doctor>(ElincloudServer.Common.ApplicationEx.MultiHospitalProcName, typeof(Doctor).Name, where);
                accounts.ForEach(p => doctors.Add(new Doctor() { ID = p.ID, LoginName = p.LoginName }));

                SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
                Transaction trans = instance.GetTransaction();
                //这里再次加loginname相等的判断是为了解决登录名要区分大小写的问题
                Doctor doctor = doctors.FirstOrDefault(p => p.LoginName == user.LoginName);
                Doctor userdoctor = doctors.FirstOrDefault(p => p.Name == user.UserName);
                if (userdoctor != null)
                {
                    throw new ServiceException("W00000173");
                }
                if (doctor == null)
                {
                    #region 医院库doctor表中添加用户信息

                    doctor = new Doctor()
                    {
                        ID = instance.GetID<Doctor>(6, hospitalGroupID),
                        Name = user.UserName,
                        LoginName = user.LoginName,
                        Password = user.Password,
                        TelePhone = user.TelePhone,
                        Sex = (int)user.Sex,
                        Remark = user.Remark,
                        Birthday = user.Birthday,
                        Email = user.Email,
                        HeadPortraitRelaFileName = user.HeadPortraitRelaFileName,
                        DiagnosisSignatureRelaFileName = user.DiagnosisSignatureRelaFileName,
                        JobTitle = user.JobTitle,
                        Address = user.Address,
                        UserState = (int)user.UserState,
                        HospitalID = hospitalID,
                        CreateUserID = operationUserID,
                        CreateDate = DateTime.Now,
                        UpdateUserID = operationUserID,
                        UpdateDate = DateTime.Now,
                    };
                    trans.Insert<Doctor>(doctor);

                    #endregion

                    #region 添加角色

                    DoctorRoleRela dr = new DoctorRoleRela()
                    {
                        DoctorID = doctor.ID,
                        RoleID = roleID
                    };
                    trans.Insert(dr);

                    #endregion

                    #region 添加机构

                    DoctorOrganizationRela userOrgRela = new DoctorOrganizationRela()
                    {
                        DoctorID = doctor.ID,
                        OrganizationID = orgID,
                    };
                    trans.Insert(userOrgRela);

                    #endregion

                    #region 添加数据埋点

                    RegisterUserRecord rur = new RegisterUserRecord()
                    {
                        RegisterUserID = doctor.ID,
                        RegisterURL = registerURL,
                        CreateUserID = operationUserID,
                        CreateDate = DateTime.Now,
                        UpdateDate = DateTime.Now,
                        UpdateUserID = operationUserID
                    };
                    trans.Insert<RegisterUserRecord>(rur);

                    #endregion

                    if (!trans.Commit())
                    {
                        throw new ServiceException("W00000418");
                    }
                }
                else
                {
                    throw new ServiceException("W00000174");
                }

            }
            else
            {
                throw new ServiceException("W00000175");
            }
        }

        public void CheckVerificationCode(string telePhone, string verificationCode)
        {
            if (!string.IsNullOrWhiteSpace(telePhone) && !string.IsNullOrWhiteSpace(verificationCode))
            {
                string result = ElincloudServer.Common.Utils.SMSUtil.CheckVerificationCode(telePhone, verificationCode);
                if (result != string.Empty)
                {
                    throw new ServiceException(result);
                }
            }
            else
            {
                throw new ServiceException("W00000370");
            }
        }

        #endregion

        #endregion

        #region 角色管理

        /// <summary>
        /// 获取医院的角色信息
        /// </summary>
        /// <param name="hospitalID"></param>
        /// <returns></returns>
        public List<RoleInfo> GetHospitalRoles(string hospitalID, string hospitalGroupID, string searchStr)
        {
            List<RoleInfo> roleInfos = new List<RoleInfo>();
            SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
            Expression<Func<Role, bool>> whereExpression = p => p.HospitalID == hospitalID;
            if (!string.IsNullOrEmpty(searchStr))
            {
                whereExpression = whereExpression.And(x => (x.Name.Contains(searchStr)));
            }
            List<Role> roles = instance.GetMuch<Role>(whereExpression);
            foreach (var role in roles)
            {
                RoleInfo roleInfo = new RoleInfo()
                {
                    ID = role.ID,
                    Name = role.Name,
                    Remark = role.Remark
                };
                roleInfos.Add(roleInfo);
            }
            return roleInfos;
        }

        /// <summary>
        /// 获取不在医院角色下的用户
        /// </summary>
        /// <param name="hospitalID"></param>
        /// <param name="pageIndex"></param>
        /// <param name="pageSize"></param>
        /// <returns></returns>
        public List<SimpleUserInfo> GetSimpleUsersWithoutAnyRole(string hospitalID, string hospitalGroupID)
        {
            List<SimpleUserInfo> userInfos = new List<SimpleUserInfo>();

            SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
            //医生和角色关联表
            List<DoctorRoleRela> doctorOrganizationRelas = instance.GetMuch<DoctorRoleRela>();
            List<string> doctorIDs = doctorOrganizationRelas.Select(x => x.DoctorID).ToList();

            List<Doctor> doctors = instance.GetMuch<Doctor>(p => p.HospitalID == hospitalID && p.UserState == (int)SysStatus.启用);
            List<Doctor> infos = doctors.Where(doc => !doctorIDs.Contains(doc.ID)).ToList();
            foreach (var item in infos)
            {
                userInfos.Add(new SimpleUserInfo()
                {
                    ID = item.ID,
                    UserName = item.Name
                });
            }
            return userInfos;
        }

        public Page<RoleInfo> GetHospitalRolesByPage(string hospitalID, string hospitalGroupID, string searchStr, int pageIndex, int pageSize)
        {
            List<RoleInfo> roleInfos = new List<RoleInfo>();
            if (pageSize > 0 && pageIndex > 0)
            {
                SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
                Expression<Func<Role, bool>> whereExpression = p => p.HospitalID == hospitalID;
                if (!string.IsNullOrEmpty(searchStr))
                {
                    whereExpression = whereExpression.And(x => (x.Name.Contains(searchStr)));
                }

                Page<Role> roles = instance.GetMuchByPage<Role>(new DBPage(pageSize, pageIndex), whereExpression);
                foreach (var role in roles.Content)
                {
                    RoleInfo roleInfo = new RoleInfo()
                    {
                        ID = role.ID,
                        Name = role.Name,
                        Remark = role.Remark
                    };
                    roleInfos.Add(roleInfo);
                }
                return new Page<RoleInfo>(pageSize, pageIndex, roles.TotalElements, roleInfos);
            }
            return new Page<RoleInfo>(pageSize, pageIndex, 0, roleInfos);

        }

        public string AddRole(PermissionInfo pi, string hospitalID, string hospitalGroupID, RoleInfo roleInfo)
        {
            if (roleInfo != null)
            {
                SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
                Role r = instance.GetOne<Role>(p => p.Name == roleInfo.Name && p.HospitalID == hospitalID);
                if (r != null)
                {
                    throw new ServiceException("W00000206");
                }
                //新增角色
                Role role = new Role()
                {
                    ID = instance.GetID<Role>(),
                    Name = roleInfo.Name,
                    Remark = roleInfo.Remark,
                    HospitalID = hospitalID,
                    CreateUserID = pi.UserID,
                    CreateDate = DateTime.Now,
                    UpdateUserID = pi.UserID,
                    UpdateDate = DateTime.Now
                };
                instance.Insert<Role>(role);
                return role.ID;
            }
            else
            {
                throw new ServiceException("W00000207");
            }
        }

        public void UpdateRole(PermissionInfo pi, string hospitalID, string hospitalGroupID, RoleInfo roleInfo)
        {
            if (roleInfo != null)
            {
                SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
                //根据角色ID查询角色信息
                var currentRole = instance.GetOne<Role>(r => r.ID == roleInfo.ID);
                if (currentRole != null)
                {

                    Role r = instance.GetOne<Role>(p => p.Name == roleInfo.Name && p.HospitalID == hospitalID && p.ID != roleInfo.ID);
                    if (r != null)
                    {
                        throw new ServiceException("W00000206");
                    }

                    currentRole.Name = roleInfo.Name;
                    currentRole.Remark = roleInfo.Remark;
                    currentRole.UpdateUserID = pi.UserID;
                    currentRole.UpdateDate = DateTime.Now;
                    //更新角色
                    instance.Update<Role>(currentRole, role => role.ID == currentRole.ID, Role._Name, Role._Remark, Role._UpdateDate, Role._UpdateUserID);
                }
                else
                {
                    throw new ServiceException("W00000208");
                }
            }
            else
            {
                throw new ServiceException("W00000207");
            }
        }

        public void DeleteRole(PermissionInfo pi, string hospitalGroupID, string roleID)
        {
            SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);

            int dotCount = instance.GetCount<DoctorRoleRela>(p => p.RoleID == roleID);
            if (dotCount > 0)
            {
                throw new ServiceException("W00000210");
            }

            var role = instance.GetOne<Role>(r => r.ID == roleID);
            if (role != null)
            {
                Transaction basicTran = instance.GetTransaction();
                basicTran.Delete<DoctorRoleRela>(d_r => d_r.RoleID == role.ID);
                basicTran.Delete<Role>(d => d.ID == roleID);
                if (!basicTran.Commit())
                {
                    throw new ServiceException("W00000211");
                }
            }
        }

        public void AddDoctorRoleRelas(PermissionInfo pi, string hospitalGroupID, string roleID, List<string> userIds)
        {
            if (string.IsNullOrWhiteSpace(roleID))
            {
                throw new ServiceException("W00000168");
            }
            if (userIds == null && userIds.Count == 0)
            {
                throw new ServiceException("W00000168");
            }

            SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
            Transaction tran = instance.GetTransaction();
            tran.Delete<DoctorRoleRela>(x => x.RoleID == roleID);

            DateTime dateTime = DateTime.Now;
            foreach (var userid in userIds)
            {
                DoctorRoleRela dr = new DoctorRoleRela()
                {
                    DoctorID = userid,
                    RoleID = roleID
                };
                tran.Insert(dr);
            }
            if (!tran.Commit())
            {
                throw new ServiceException("W00000214");
            }
        }

        #endregion

        #region 机构管理
        /// <summary>
        /// 获取医院分组下的组织机构
        /// </summary>
        /// <returns></returns>
        public List<OrganizationInfo> GetOrganizationInfoByHospitalGroupID(string hospitalGroupID)
        {
            List<OrganizationInfo> infos = new List<OrganizationInfo>();
            SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
            List<Organization> list = instance.GetMuch<Organization>();
            infos = list.Select(i =>
            {
                return new OrganizationInfo()
                {
                    ID = i.ID,
                    Name = i.Name,
                    Code = i.Code,
                    PID = i.PID,
                    Remark = i.Remark,
                    HospitalID = i.HospitalID
                };
            }).ToList();
            return infos;
        }

        /// <summary>
        /// 获取医院的机构信息
        /// </summary>
        /// <param name="hospitalID"></param>
        /// <returns></returns>
        public List<OrganizationInfo> GetHospitalOrganizations(string hospitalID, string hospitalGroupID, string searchStr)
        {
            List<OrganizationInfo> organizationInfos = new List<OrganizationInfo>();
            SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
            Expression<Func<Organization, bool>> whereExpression = p => p.HospitalID == hospitalID;
            if (!string.IsNullOrEmpty(searchStr))
            {
                whereExpression = whereExpression.And(x => x.Name.Contains(searchStr));
            }
            List<Organization> organizations = instance.GetMuch<Organization>(whereExpression);
            foreach (var organization in organizations)
            {
                OrganizationInfo organizationInfo = new OrganizationInfo()
                {
                    ID = organization.ID,
                    Name = organization.Name,
                    Code = organization.Code,
                    PID = organization.PID,
                    Remark = organization.Remark
                };
                organizationInfos.Add(organizationInfo);
            }
            return organizationInfos;
        }

        /// <summary>
        ///  获取不在医院机构下的用户
        /// </summary>
        /// <param name="hospitalID"></param>
        /// <param name="pageIndex"></param>
        /// <param name="pageSize"></param>
        /// <returns></returns>
        public List<SimpleUserInfo> GetSimpleUsersWithoutAnyOrganization(string hospitalID, string hospitalGroupID)
        {
            List<SimpleUserInfo> userInfos = new List<SimpleUserInfo>();
            SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
            //医生和机构关联表
            List<DoctorOrganizationRela> doctorOrganizationRelas = instance.GetMuch<DoctorOrganizationRela>();
            List<string> doctorIDs = doctorOrganizationRelas.Select(x => x.DoctorID).ToList();
            //Expression<Func<Doctor, bool>> whereExpression = x=> !doctorIDs.Contains( x.ID);
            List<Doctor> doctors = instance.GetMuch<Doctor>(p => p.HospitalID == hospitalID);
            List<Doctor> infos = doctors.Where(doc => !doctorIDs.Contains(doc.ID)).ToList();
            foreach (var item in infos)
            {
                userInfos.Add(new SimpleUserInfo()
                {
                    ID = item.ID,
                    UserName = item.Name
                });
            }
            return userInfos;
        }

        public OrganizationInfo GetOrganizationInfoByUserID(string hospitalGroupID, string userID)
        {
            OrganizationInfo organizationInfo = new OrganizationInfo();
            SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
            DoctorOrganizationRela doctorOrganizationRela = instance.GetOne<DoctorOrganizationRela>(p => p.DoctorID == userID);
            if (doctorOrganizationRela != null)
            {
                Organization organization = instance.GetOne<Organization>(p => p.ID == doctorOrganizationRela.OrganizationID);
                organizationInfo.ID = organization.ID;
                organizationInfo.Name = organization.Name;
                organizationInfo.Code = organization.Code;
                organizationInfo.PID = organization.PID;
                organizationInfo.Remark = organization.Remark;
            }
            return organizationInfo;
        }

        public string AddOrganization(PermissionInfo pi, string hospitalID, string hospitalGroupID, OrganizationInfo orgInfo)
        {

            if (orgInfo != null && !string.IsNullOrEmpty(orgInfo.Name))
            {
                SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
                var orgs = instance.GetMuch<Organization>();
                if (orgs.Exists(p => p.PID == orgInfo.PID && p.Name == orgInfo.Name))
                {
                    throw new ServiceException("W00000196");
                }
                else
                {
                    //新增组织机构信息
                    Organization organization = new Organization()
                    {
                        ID = instance.GetID<Organization>(4),
                        Name = orgInfo.Name,
                        PID = string.IsNullOrEmpty(orgInfo.PID) ? ElincloudServer.Common.ApplicationEx.RootID : orgInfo.PID,
                        Remark = orgInfo.Remark,
                        Code = orgInfo.Code,
                        HospitalID = hospitalID,
                        CreateUserID = pi.UserID,
                        CreateDate = DateTime.Now,
                        UpdateUserID = pi.UserID,
                        UpdateDate = DateTime.Now
                    };
                    instance.Insert<Organization>(organization);
                    return organization.ID;
                }
            }
            else
            {
                throw new ServiceException("W00000197");
            }
        }

        public void UpdateOrganization(PermissionInfo pi, string hospitalID, string hospitalGroupID, OrganizationInfo orgInfo)
        {
            if (string.IsNullOrEmpty(orgInfo.Name) || string.IsNullOrEmpty(orgInfo.ID))
            {
                throw new ServiceException("W00000198");
            }

            SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
            var orgs = instance.GetMuch<Organization>();
            Organization organization = orgs.FirstOrDefault(p => p.ID == orgInfo.ID);
            if (organization != null)
            {
                if (orgs.Exists(p => p.PID == orgInfo.PID && p.Name == orgInfo.Name && p.ID != orgInfo.ID && orgInfo.HospitalID == hospitalID))
                {
                    throw new ServiceException("W00000199");
                }
                else
                {
                    Transaction trans = instance.GetTransaction();
                    organization.Code = orgInfo.Code;
                    organization.Name = orgInfo.Name;
                    organization.UpdateUserID = pi.UserID;
                    organization.UpdateDate = DateTime.Now;
                    organization.Remark = orgInfo.Remark;

                    //根据组织机构ID更新组织机构信息
                    trans.Update<Organization>(organization, p => p.ID == organization.ID, Organization._Code, Organization._Name, Organization._Remark, Organization._UpdateDate, Organization._UpdateUserID);
                    if (!trans.Commit())
                    {
                        throw new ServiceException("W00000200");
                    }
                }
            }
            else
            {
                throw new ServiceException("W00000201");
            }
        }

        public void DeleteOrganization(PermissionInfo pi, string hospitalGroupID, string orgID)
        {

            SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
            //删除机构时 先判断有没有下级机构，在判断机构下有没有用户
            List<Organization> allOrgs = instance.GetMuch<Organization>();
            List<OrganizationInfo> subOrgs = CommonBL.GetSubOrganizations(orgID, allOrgs);
            if (subOrgs != null && subOrgs.Count > 0)
            {
                throw new ServiceException("W00000202");
            }

            int dorCount = instance.GetCount<DoctorOrganizationRela>(c => c.OrganizationID == orgID);
            if (dorCount > 0)
            {
                throw new ServiceException("W00000203");
            }
            instance.Delete<Organization>(p => p.ID == orgID);
        }

        /// <summary>
        /// 修改组织机构下的用户人员
        /// </summary>
        /// <param name="hospitalID"></param>
        /// <param name="orgID"></param>
        /// <param name="userIDs">用户人员ID集合</param>
        /// whz 2024年11月18日09:59:51
        public void UpdateUsersOrganizationRelas(string hospitalGroupID, string orgID, List<string> userIDs)
        {
            if (!string.IsNullOrEmpty(orgID))
            {
                SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
                Transaction tran = instance.GetTransaction();
                if (userIDs != null && userIDs.Count > 0)
                {
                    tran.Delete<DoctorOrganizationRela>(c => c.OrganizationID == orgID);
                    foreach (var userID in userIDs)
                    {
                        tran.Delete<DoctorOrganizationRela>(c => c.DoctorID == userID);
                        DoctorOrganizationRela userOrgRela = new DoctorOrganizationRela()
                        {
                            DoctorID = userID,
                            OrganizationID = orgID,
                        };
                        tran.Insert(userOrgRela);
                    }
                    if (!tran.Commit())
                    {
                        throw new ServiceException("W00000204");
                    }
                }
                else
                {
                    throw new ServiceException("W00000205");
                }
            }
            else
            {
                throw new ServiceException("W00000181");
            }
        }

        #endregion

        #region 模块管理

        /// <summary>
        /// 获取所有模块信息
        /// </summary>
        /// <returns></returns>
        public List<ModuleInfo> GetModules(string hospitalGroupID)
        {
            SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
            List<ModuleInfo> rets = new List<ModuleInfo>();
            List<Module> modules = instance.GetMuch<Module>();
            if (modules != null && modules.Count > 0)
            {
                foreach (var module in modules)
                {
                    rets.Add(new ModuleInfo()
                    {
                        ID = module.ID,
                        PID = module.PID,
                        Name = module.Name,
                        Remark = module.Remark,
                        Action = module.Action,
                        ModulePurpose = ElincloudServer.Common.Utils.EnumUtil.GetEnumValue<ModulePurposes>(module.ModulePurpose),
                        SortIndex = module.SortIndex,
                        Image = module.Image,
                        ThirdUrl = module.ThirdUrl
                    });
                }
            }
            return rets;
        }

        /// <summary>
        /// 根据角色获取模块
        /// </summary>
        /// <param name="hospitalGroupID"></param>
        /// <param name="modulePurposes"></param>
        /// <param name="roleIDs"></param>
        /// <returns></returns>
        /// <exception cref="ServiceException"></exception>
        public List<ModuleInfo> GetPermittedModulesByRoles(string hospitalGroupID, ModulePurposes? modulePurposes, params string[] roleIDs)
        {
            List<ModuleInfo> rets = new List<ModuleInfo>();
            if (roleIDs != null && roleIDs.Length > 0)
            {
                SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
                List<RoleModuleRela> roleModuleRelas = instance.GetMuch<RoleModuleRela>(p => roleIDs.ToList().Contains(p.RoleID));
                if (roleModuleRelas != null && roleModuleRelas.Count > 0)
                {
                    List<string> moduleIDs = roleModuleRelas.Select(c => c.ModuleID).ToList();
                    Expression<Func<Module, bool>> whereExpression = p => moduleIDs.Contains(p.ID);

                    //modulePurpose为空查所有，不为空按参数来查
                    if (modulePurposes != null)
                    {
                        whereExpression = whereExpression.And(p => p.ModulePurpose == (int)modulePurposes);
                    }

                    List<Module> permittedModules = instance.GetMuch<Module>(whereExpression);
                    if (permittedModules != null && permittedModules.Count > 0)
                    {
                        foreach (var module in permittedModules)
                        {
                            ModuleInfo moduleIndo = new ModuleInfo()
                            {
                                ID = module.ID,
                                Name = module.Name,
                                PID = module.PID,
                                Action = module.Action,
                                ModulePurpose = ElincloudServer.Common.Utils.EnumUtil.GetEnumValue<ModulePurposes>(module.ModulePurpose),
                                SortIndex = module.SortIndex,
                                Remark = module.Remark,
                                Image = module.Image,
                                ResourcesKey = module.ResourcesKey,
                                ThirdUrl = module.ThirdUrl
                            };
                            rets.Add(moduleIndo);
                        }
                    }
                    else
                    {
                        throw new ServiceException("W00000160");
                    }
                }
            }
            return rets;

        }

        /// <summary>
        /// 新增模块信息
        /// </summary>
        /// <param name="pi"></param>
        /// <param name="moduleInfo"></param>
        /// <returns></returns>
        public void AddModule(PermissionInfo pi, string hospitalGroupID, ModuleInfo moduleInfo)
        {
            if (moduleInfo != null)
            {
                SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
                Module module = instance.GetOne<Module>(c => c.Name == moduleInfo.Name && c.PID == moduleInfo.PID);
                if (module == null)
                {
                    module = new Module()
                    {
                        ID = instance.GetID<Module>(),
                        Name = moduleInfo.Name,
                        PID = string.IsNullOrEmpty(moduleInfo.PID) ? ElincloudServer.Common.ApplicationEx.RootID : moduleInfo.PID,
                        Action = moduleInfo.Action,
                        ModulePurpose = (int)moduleInfo.ModulePurpose,
                        SortIndex = moduleInfo.SortIndex,
                        CreateUserID = pi.UserID,
                        CreateDate = DateTime.Now,
                        UpdateUserID = pi.UserID,
                        UpdateDate = DateTime.Now,
                        Remark = moduleInfo.Remark,
                        Image = moduleInfo.Image,
                        ThirdUrl = moduleInfo.ThirdUrl,
                        ResourcesKey = string.Format("language_{0}", Common.Utils.PinyinUtil.GetPinyin(moduleInfo.Name))
                    };
                    instance.Insert<Module>(module);
                }
                else
                {
                    throw new ServiceException("W00000161");
                }

            }
            else
            {
                throw new ServiceException("W00000162");
            }
        }

        /// <summary>
        /// 更新模块信息
        /// </summary>
        /// <param name="moduleInfo"></param>
        public void UpdateModule(PermissionInfo pi, string hospitalGroupID, ModuleInfo moduleInfo)
        {
            if (moduleInfo != null)
            {
                SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
                var module = instance.GetOne<Module>(m => m.ID == moduleInfo.ID);
                if (module != null)
                {
                    //验证修改后的模块名称不能和已经存在的模块重名
                    Module m = instance.GetOne<Module>(p => p.Name == moduleInfo.Name && p.ID != moduleInfo.ID && p.PID == moduleInfo.PID);
                    if (m != null)
                    {
                        throw new ServiceException("W00000161");
                    }

                    module.Name = moduleInfo.Name;
                    module.Action = moduleInfo.Action;
                    module.ModulePurpose = (int)moduleInfo.ModulePurpose;
                    module.SortIndex = moduleInfo.SortIndex;
                    module.UpdateUserID = pi.UserID;
                    module.UpdateDate = DateTime.Now;
                    module.Remark = moduleInfo.Remark;
                    module.Image = moduleInfo.Image;
                    module.ThirdUrl = moduleInfo.ThirdUrl;
                    instance.Update<Module>(module, p => p.ID == module.ID,
                        Module._Name,
                        Module._Action,
                        Module._ModulePurpose,
                        Module._SortIndex,
                        Module._UpdateDate,
                        Module._UpdateUserID,
                        Module._Remark,
                        Module._Image,
                        Module._ThirdUrl);
                }
                else
                {
                    throw new ServiceException("W00000163");
                }

            }
            else
            {
                throw new ServiceException("W00000162");
            }
        }

        /// <summary>
        /// 根据模块ID删除模块信息
        /// </summary>
        /// <param name="moduleInfo">模块信息</param>
        public void DeleteModule(string hospitalGroupID, string moduleID)
        {
            SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
            //验证要删除的模块下还有子模块时不允许删除
            int childModuleCount = instance.GetCount<Module>(p => p.PID == moduleID);
            if (childModuleCount > 0)
            {
                throw new ServiceException("W00000164");
            }
            Transaction tran = instance.GetTransaction();
            //删除当前模块
            tran.Delete<Module>(p => p.ID == moduleID);
            //删除所有医院模块权限信息
            tran.Delete<RoleModuleRela>(p => p.ModuleID == moduleID);
            //提交删除
            if (!tran.Commit())
            {
                throw new ServiceException("W00000165");
            }
        }

        /// <summary>
        /// 根据角色信息维护权限信息
        /// </summary>
        /// <param name="roleID"></param>
        public void UpdatePermisionByRole(string hospitalGroupID, string roleID, List<string> moduleIDs)
        {
            if (!string.IsNullOrEmpty(roleID))
            {
                if (moduleIDs != null && moduleIDs.Count > 0)
                {
                    SqlAccess instance = DBServices.GetHospitalDBInstance(hospitalGroupID);
                    Transaction tran = instance.GetTransaction();
                    List<RoleModuleRela> relas = new List<RoleModuleRela>();
                    foreach (var mID in moduleIDs)
                    {
                        relas.Add(new RoleModuleRela()
                        {
                            ModuleID = mID,
                            RoleID = roleID,
                        });
                    }
                    tran.Delete<RoleModuleRela>(p => p.RoleID == roleID);
                    tran.Insert<RoleModuleRela>(relas.ToArray());
                    if (!tran.Commit())
                    {
                        throw new ServiceException("W00000166");
                    }
                }
                else
                {
                    throw new ServiceException("W00000167");
                }

            }
            else
            {
                throw new ServiceException("W00000168");
            }
        }

        #endregion



        #region 运维用户管理
        /// <summary>
        /// 根据用户ID获取运维用户
        /// </summary>
        /// <param name="userID"></param>
        /// <returns></returns>
        /// <exception cref="ServiceException"></exception>
        public UserInfo GetAdminUserByID(string userID)
        {
            Admin account = DBServices.ElincloudBasicInstance.GetOne<Admin>(p => p.ID == userID);
            if (account != null)
            {
                UserInfo user = new UserInfo()
                {
                    ID = account.ID,
                    UserName = account.UserName,
                    LoginName = account.LoginName,
                    Password = account.Password,
                    UserState = ElincloudServer.Common.Utils.EnumUtil.GetEnumValue<SysStatus>(account.UserState),
                    TelePhone = account.TelePhone,
                    Email = account.Email,
                    IsAdmin = true
                };
                return user;
            }
            else
            {
                throw new ServiceException("W00000169");
            }
        }

        /// <summary>
        /// 获取所有运维用户
        /// </summary>
        /// <returns></returns>
        public List<UserInfo> GetAdminUsers()
        {
            List<UserInfo> userInfos = new List<UserInfo>();
            List<Admin> list = DBServices.ElincloudBasicInstance.GetMuch<Admin>(p => p.LoginName != Common.ApplicationEx.SuperAdmin);
            foreach (var doc in list)
            {
                UserInfo user = new UserInfo()
                {
                    ID = doc.ID,
                    UserName = doc.UserName,
                    LoginName = doc.LoginName,
                    Password = doc.Password,
                    UserState = ElincloudServer.Common.Utils.EnumUtil.GetEnumValue<SysStatus>(doc.UserState),
                    TelePhone = doc.TelePhone,
                    Email = doc.Email,
                    IsAdmin = true,
                };
                userInfos.Add(user);
            }
            return userInfos;
        }

        /// <summary>
        /// 分页获取运维用户列表
        /// </summary>
        /// <param name="pi"></param>
        /// <param name="priority"></param>
        /// <returns></returns>
        public Entity.Page<UserInfo> GetAdminUsersByPage(int pageIndex, int pageSize)
        {
            List<UserInfo> userInfos = new List<UserInfo>();
            Entity.Page<Admin> pageAccount = DBServices.ElincloudBasicInstance.GetMuchByPage<Admin>(new DBPage(pageSize, pageIndex), p => p.LoginName != Common.ApplicationEx.SuperAdmin);
            foreach (var doc in pageAccount.Content)
            {
                UserInfo user = new UserInfo()
                {
                    ID = doc.ID,
                    UserName = doc.UserName,
                    LoginName = doc.LoginName,
                    Password = doc.Password,
                    UserState = ElincloudServer.Common.Utils.EnumUtil.GetEnumValue<SysStatus>(doc.UserState),
                    TelePhone = doc.TelePhone,
                    Email = doc.Email,
                    IsAdmin = true,
                };
                userInfos.Add(user);
            }
            return new Entity.Page<UserInfo>(pageSize, pageIndex, pageAccount.TotalElements, userInfos);
        }

        /// <summary>
        /// 根据运维角色ID获取运维用户列表
        /// </summary>
        /// <param name="roleID"></param>
        /// <returns></returns>
        public List<UserInfo> GetAdminUsersByRole(string roleID)
        {
            List<UserInfo> rets = new List<UserInfo>();
            List<AdminRoleRela> roleRelas = DBServices.ElincloudBasicInstance.GetMuch<AdminRoleRela>(p => p.RoleID == roleID);
            if (roleRelas != null)
            {
                List<string> adminIDs = roleRelas.Select(c => c.AdminID).ToList();
                if (adminIDs != null)
                {
                    List<Admin> admins = DBServices.ElincloudBasicInstance.GetMuch<Admin>(p => adminIDs.Contains(p.ID) && p.UserState == (int)SysStatus.启用);
                    if (admins != null)
                    {
                        foreach (var item in admins)
                        {
                            UserInfo ret = new UserInfo()
                            {
                                ID = item.ID,
                                UserName = item.UserName,
                                LoginName = item.LoginName,
                                Password = item.Password,
                                UserState = ElincloudServer.Common.Utils.EnumUtil.GetEnumValue<SysStatus>(item.UserState),
                                TelePhone = item.TelePhone,
                                Email = item.Email,
                                IsAdmin = true
                            };
                            rets.Add(ret);
                        }
                    }
                }
            }
            return rets;
        }

        /// <summary>
        ///  根据运维用户角色ID分页获取运维用户列表
        /// </summary>
        /// <param name="roleID"></param>
        /// <param name="pageIndex"></param>
        /// <param name="pageSize"></param>
        /// <returns></returns>
        public Page<UserInfo> GetAdminUsersByRoleByPage(string roleID, int pageIndex, int pageSize)
        {
            List<UserInfo> rets = new List<UserInfo>();
            List<AdminRoleRela> roleRelas = DBServices.ElincloudBasicInstance.GetMuch<AdminRoleRela>(p => p.RoleID == roleID);
            if (roleRelas != null)
            {
                List<string> ids = roleRelas.Select(c => c.AdminID).ToList();
                if (ids != null)
                {
                    Page<Admin> entities = DBServices.ElincloudBasicInstance.GetMuchByPage<Admin>(new DBPage(pageSize, pageIndex), p => ids.Contains(p.ID));
                    if (entities.Content != null)
                    {
                        foreach (var item in entities.Content)
                        {
                            UserInfo ret = new UserInfo()
                            {
                                ID = item.ID,
                                UserName = item.UserName,
                                LoginName = item.LoginName,
                                Password = item.Password,
                                UserState = ElincloudServer.Common.Utils.EnumUtil.GetEnumValue<SysStatus>(item.UserState),
                                TelePhone = item.TelePhone,
                                Email = item.Email,
                                IsAdmin = true
                            };
                            rets.Add(ret);
                        }
                        return new Entity.Page<UserInfo>(pageSize, pageIndex, entities.TotalElements, rets); ;
                    }
                }
            }
            return new Entity.Page<UserInfo>(pageSize, pageIndex, 0, rets); ;

        }

        /// <summary>
        /// 设置管理员用户启用、停用状态
        /// </summary>
        /// <param name="pi"></param>
        /// <param name="userID"></param>
        /// <param name="state"></param>
        /// <exception cref="ServiceException"></exception>
        public void UpdateAdminUserStatus(PermissionInfo pi, string userID, SysStatus state)
        {
            Admin account = DBServices.ElincloudBasicInstance.GetOne<Admin>(p => p.ID == userID);
            if (account != null)
            {
                account.UserState = (int)state;
                account.UpdateDate = DateTime.Now;
                account.UpdateUserID = pi.UserID;
                DBServices.ElincloudBasicInstance.Update<Admin>(account, p => p.ID == userID, Admin._UserState, Admin._UpdateDate, Admin._UpdateUserID);
            }
            else
            {
                throw new ServiceException("W00000169");
            }
        }

        /// <summary>
        /// 新增运维用户
        /// </summary>
        /// <param name="user"></param>
        /// <returns>ID</returns>
        public void AddAdminUser(PermissionInfo pi, UserInfo user)
        {
            if (user != null)
            {
                if (!ElincloudServer.Common.Utils.StringUtil.LoginNameValid(user.LoginName))
                {
                    throw new ServiceException("W00000172");
                }
                //添加用户时需要验证所有已存在用户的登录名是否存在
                List<Admin> accounts = DBServices.ElincloudBasicInstance.GetMuch<Admin>(p => p.LoginName == user.LoginName);
                string where = DBServices.ElincloudBasicInstance.GetWhereByExpression<Doctor>(p => p.LoginName == user.LoginName || p.Name == user.UserName);
                List<Doctor> doctors = DBServices.ElincloudBasicInstance.GetMuchByProcedure<Doctor>(ElincloudServer.Common.ApplicationEx.MultiHospitalProcName, typeof(Doctor).Name, where);
                accounts.ForEach(p => doctors.Add(new Doctor() { ID = p.ID, LoginName = p.LoginName }));

                //这里再次加loginname相等的判断是为了解决登录名要区分大小写的问题
                Doctor doctor = doctors.FirstOrDefault(p => p.LoginName == user.LoginName);
                Doctor userdoctor = doctors.FirstOrDefault(p => p.Name == user.UserName);
                if (userdoctor != null)
                {
                    throw new ServiceException("W00000173");
                }

                if (doctor == null)
                {
                    #region Basic库account表中添加用户信息

                    Admin account = new Admin()
                    {
                        ID = DBServices.ElincloudBasicInstance.GetID<Admin>(6, "Admin"),
                        UserName = user.UserName,
                        LoginName = user.LoginName,
                        Password = user.Password,
                        UserState = (int)user.UserState,
                        TelePhone = user.TelePhone,
                        Email = user.Email,
                        CreateDate = DateTime.Now,
                        CreateUserID = pi.UserID,
                        UpdateDate = DateTime.Now,
                        UpdateUserID = pi.UserID,
                    };
                    DBServices.ElincloudBasicInstance.Insert<Admin>(account);

                    #endregion
                }
                else
                {
                    throw new ServiceException("W00000174");
                }
            }
            else
            {
                throw new ServiceException("W00000175");
            }
        }

        /// <summary>
        /// 更新运维用户
        /// </summary>
        /// <param name="user"></param>
        public void UpdateAdminUser(PermissionInfo pi, UserInfo user)
        {
            if (user != null)
            {
                List<Admin> users = DBServices.ElincloudBasicInstance.GetMuch<Admin>();
                Admin account = users.FirstOrDefault(i => i.ID == user.ID);
                if (account != null)
                {
                    if (users.FindAll(i => i.UserName == user.UserName && i.ID != user.ID).Count > 0)
                    {
                        throw new ServiceException("W00000182");
                    }

                    List<Transaction> trans = new List<Transaction>();

                    #region 修改通用帐号信息

                    account.UserName = user.UserName;
                    account.TelePhone = user.TelePhone;
                    account.Email = user.Email;
                    account.Password = user.Password;
                    DBServices.ElincloudBasicInstance.Update<Admin>(account, p => p.ID == user.ID, Admin._UserName, Admin._TelePhone, Admin._Email, Admin._Password);

                    #endregion

                }
                else
                {
                    throw new ServiceException("W00000169");
                }
            }
        }

        /// <summary>
        /// 修改运维用户的角色
        /// </summary>
        /// <param name="pi"></param>
        /// <param name="userID">要添加的用户ID</param>
        /// <param name="roleIDs">角色列表</param>
        /// <returns></returns>
        public void UpdateAdminUserRoleRelas(string userID, List<string> roleIDs)
        {
            if (roleIDs.Count > 0)
            {
                SqlAccess instance = DBServices.ElincloudBasicInstance;
                Admin entity = instance.GetOne<Admin>(p => p.ID == userID);
                if (entity != null)
                {
                    Transaction tran = instance.GetTransaction();
                    List<AdminRoleRela> relas = new List<AdminRoleRela>();
                    foreach (var roleID in roleIDs)
                    {
                        relas.Add(new AdminRoleRela()
                        {
                            AdminID = entity.ID,
                            RoleID = roleID,
                        });
                    }
                    tran.Delete<AdminRoleRela>(c => c.AdminID == entity.ID);
                    tran.Insert<AdminRoleRela>(relas.ToArray());
                    if (!tran.Commit())
                    {
                        throw new ServiceException("W00000176");
                    }
                }
                else
                {
                    throw new ServiceException("W00000177");
                }
            }
            else
            {
                throw new ServiceException("W00000168");
            }
        }
        #endregion

        #region 运维角色管理

        /// <summary>
        /// 获取运维角色
        /// </summary>
        /// <param name="searchStr"></param>
        /// <returns></returns>
        public List<RoleInfo> GetAdminRoles(string searchStr)
        {
            List<RoleInfo> roleInfos = new List<RoleInfo>();

            Expression<Func<AdminRole, bool>> whereExpression = null;
            if (!string.IsNullOrEmpty(searchStr))
            {
                whereExpression = x => (x.Name.Contains(searchStr));
            }
            List<AdminRole> roles = DBServices.ElincloudBasicInstance.GetMuch<AdminRole>(whereExpression);
            foreach (var role in roles)
            {
                RoleInfo roleInfo = new RoleInfo()
                {
                    ID = role.ID,
                    Name = role.Name,
                    Remark = role.Remark
                };
                roleInfos.Add(roleInfo);
            }
            return roleInfos;
        }

        /// <summary>
        /// 分页获取运维角色
        /// </summary>
        /// <param name="searchStr"></param>
        /// <param name="pageIndex"></param>
        /// <param name="pageSize"></param>
        /// <returns></returns>
        public Page<RoleInfo> GetAdminRolesByPage(string searchStr, int pageIndex, int pageSize)
        {
            List<RoleInfo> roleInfos = new List<RoleInfo>();
            if (pageSize > 0 && pageIndex > 0)
            {
                Expression<Func<AdminRole, bool>> whereExpression = null;
                if (!string.IsNullOrEmpty(searchStr))
                {
                    whereExpression = x => (x.Name.Contains(searchStr));
                }

                Page<AdminRole> roles = DBServices.ElincloudBasicInstance.GetMuchByPage<AdminRole>(new DBPage(pageSize, pageIndex), whereExpression);
                foreach (var role in roles.Content)
                {
                    RoleInfo roleInfo = new RoleInfo()
                    {
                        ID = role.ID,
                        Name = role.Name,
                        Remark = role.Remark
                    };
                    roleInfos.Add(roleInfo);
                }
                return new Page<RoleInfo>(pageSize, pageIndex, roles.TotalElements, roleInfos);
            }
            return new Page<RoleInfo>(pageSize, pageIndex, 0, roleInfos);
        }

        /// <summary>
        /// 获取不在运维角色下的运维用户
        /// </summary>
        /// <returns></returns>
        public List<SimpleUserInfo> GetSimpleAdminUsersWithoutAnyRole()
        {
            List<SimpleUserInfo> userInfos = new List<SimpleUserInfo>();

            List<AdminRoleRela> doctorOrganizationRelas = DBServices.ElincloudBasicInstance.GetMuch<AdminRoleRela>();
            List<string> ids = doctorOrganizationRelas.Select(x => x.AdminID).ToList();

            List<Admin> doctors = DBServices.ElincloudBasicInstance.GetMuch<Admin>();
            List<Admin> infos = doctors.Where(p => !ids.Contains(p.ID) && p.LoginName != Common.ApplicationEx.SuperAdmin).ToList();
            foreach (var item in infos)
            {
                userInfos.Add(new SimpleUserInfo()
                {
                    ID = item.ID,
                    UserName = item.UserName
                });
            }
            return userInfos;
        }

        /// <summary>
        /// 添加运维角色
        /// </summary>
        /// <param name="pi"></param>
        /// <param name="roleInfo"></param>
        /// <returns></returns>
        /// <exception cref="ServiceException"></exception>
        public string AddAdminRole(PermissionInfo pi, RoleInfo roleInfo)
        {
            if (roleInfo != null)
            {
                SqlAccess instance = DBServices.ElincloudBasicInstance;
                AdminRole r = instance.GetOne<AdminRole>(p => p.Name == roleInfo.Name);
                if (r != null)
                {
                    throw new ServiceException("W00000206");
                }

                //新增角色
                AdminRole role = new AdminRole()
                {
                    ID = instance.GetID<AdminRole>(),
                    Name = roleInfo.Name,
                    Remark = roleInfo.Remark,
                    CreateUserID = pi.UserID,
                    CreateDate = DateTime.Now,
                    UpdateUserID = pi.UserID,
                    UpdateDate = DateTime.Now
                };
                instance.Insert<AdminRole>(role);
                return role.ID;
            }
            else
            {
                throw new ServiceException("W00000207");
            }
        }

        /// <summary>
        /// 修改运维角色
        /// </summary>
        /// <param name="pi"></param>
        /// <param name="role"></param>
        /// <exception cref="ServiceException"></exception>
        public void UpdateAdminRole(PermissionInfo pi, RoleInfo role)
        {
            if (role != null)
            {
                SqlAccess instance = DBServices.ElincloudBasicInstance;
                //根据角色ID查询角色信息
                var currentRole = instance.GetOne<AdminRole>(r => r.ID == role.ID);
                if (currentRole != null)
                {
                    AdminRole r = instance.GetOne<AdminRole>(p => p.Name == role.Name && p.ID != role.ID);
                    if (r != null)
                    {
                        throw new ServiceException("W00000206");
                    }

                    currentRole.Name = role.Name;
                    currentRole.Remark = role.Remark;
                    currentRole.UpdateUserID = pi.UserID;
                    currentRole.UpdateDate = DateTime.Now;
                    //更新角色
                    instance.Update<AdminRole>(currentRole, role => role.ID == currentRole.ID, AdminRole._Name, AdminRole._Remark, AdminRole._UpdateDate, AdminRole._UpdateUserID);
                }
                else
                {
                    throw new ServiceException("W00000208");
                }
            }
            else
            {
                throw new ServiceException("W00000207");
            }
        }

        /// <summary>
        /// 删除运维角色
        /// </summary>
        /// <param name="pi"></param>
        /// <param name="roleID"></param>
        /// <exception cref="ServiceException"></exception>
        public void DeleteAdminRole(PermissionInfo pi, string roleID)
        {
            SqlAccess instance = DBServices.ElincloudBasicInstance;

            int dotCount = instance.GetCount<AdminRoleRela>(p => p.RoleID == roleID);
            if (dotCount > 0)
            {
                throw new ServiceException("W00000210");
            }

            var role = instance.GetOne<AdminRole>(r => r.ID == roleID);
            if (role != null)
            {
                Transaction basicTran = instance.GetTransaction();
                basicTran.Delete<AdminRoleRela>(d_r => d_r.RoleID == role.ID);
                basicTran.Delete<AdminRole>(d => d.ID == roleID);
                if (!basicTran.Commit())
                {
                    throw new ServiceException("W00000211");
                }
            }
        }

        /// <summary>
        /// 给运维角色下添加运维用户
        /// </summary>
        /// <param name="pi"></param>
        /// <param name="userIds"></param>
        /// <exception cref="ServiceException"></exception>
        public void AddAdminRoleRelas(PermissionInfo pi, string roleID, List<string> userIds)
        {
            if (userIds == null && userIds.Count == 0)
            {
                throw new ServiceException("W00000168");
            }

            SqlAccess instance = DBServices.ElincloudBasicInstance;
            Transaction tran = instance.GetTransaction();
            tran.Delete<AdminRoleRela>(x => x.RoleID == roleID);

            DateTime dateTime = DateTime.Now;
            foreach (var userid in userIds)
            {
                AdminRoleRela dr = new AdminRoleRela()
                {
                    AdminID = userid,
                    RoleID = roleID
                };
                tran.Insert(dr);
            }
            if (!tran.Commit())
            {
                throw new ServiceException("W00000214");
            }
        }

        /// <summary>
        /// 删除运维角色下的用戶
        /// </summary>
        /// <param name="userID"></param>
        /// <param name="roleID"></param>
        /// <returns></returns>
        public void DeleteAdminRoleUser(string userID, string roleID)
        {
            if (!string.IsNullOrEmpty(roleID))
            {
                if (!string.IsNullOrEmpty(userID))
                {
                    SqlAccess instance = DBServices.ElincloudBasicInstance;
                    Admin doctor = instance.GetOne<Admin>(p => p.ID == userID);
                    if (doctor != null)
                    {
                        instance.Delete<AdminRoleRela>(c => c.AdminID == userID && c.RoleID == roleID);
                    }
                    else
                    {
                        throw new ServiceException("W00000177");
                    }
                }
                else
                {
                    throw new ServiceException("W00000185");
                }
            }
            else
            {
                throw new ServiceException("W00000168");
            }
        }


        public List<HospitalInfo> GetAdminHospitalInfos(string adminID)
        {
            List<HospitalInfo> hospitalInfos = new List<HospitalInfo>();
            Admin admin = DBServices.ElincloudBasicInstance.GetOne<Admin>(p => p.ID == adminID);
            if (admin.LoginName != Common.ApplicationEx.SuperAdmin)
            {
                List<AdminRoleRela> adminRoleRelas = DBServices.ElincloudBasicInstance.GetMuch<AdminRoleRela>(p => p.AdminID == adminID);
                if (adminRoleRelas != null && adminRoleRelas.Count > 0)
                {
                    List<string> roleIDs = adminRoleRelas.Select(p => p.RoleID).ToList();
                    List<AdminRoleHospitalRela> adminRoleHospitalRelas = DBServices.ElincloudBasicInstance.GetMuch<AdminRoleHospitalRela>(p => roleIDs.Contains(p.RoleID));
                    if (adminRoleHospitalRelas != null && adminRoleHospitalRelas.Count > 0)
                    {
                        hospitalInfos = GetHospitals();
                        hospitalInfos = hospitalInfos.FindAll(p => adminRoleHospitalRelas.Exists(c => c.HospitalID == p.ID));
                    }
                }
            }
            else
            {
                hospitalInfos = GetHospitals();
            }
            return hospitalInfos;
        }


        #endregion

        #region 运维模块管理
        /// <summary>
        /// 获取所有运维模块信息
        /// </summary>
        /// <returns></returns>
        public List<ModuleInfo> GetPermittedAdminModulesByLoginName(string loginName)
        {
            List<AdminModule> modules = new List<AdminModule>();
            if (loginName != Common.ApplicationEx.SuperAdmin)
            {
                Admin admin = DBServices.ElincloudBasicInstance.GetOne<Admin>(p => p.LoginName == loginName);
                if (admin != null)
                {
                    List<AdminRoleRela> adminRoleRelas = DBServices.ElincloudBasicInstance.GetMuch<AdminRoleRela>(p => p.AdminID == admin.ID);
                    if (adminRoleRelas != null && adminRoleRelas.Count > 0)
                    {
                        List<string> roleIDs = adminRoleRelas.Select(p => p.RoleID).ToList();
                        List<AdminRoleModuleRela> adminRoleModuleRelas = DBServices.ElincloudBasicInstance.GetMuch<AdminRoleModuleRela>(p => roleIDs.Contains(p.RoleID));
                        if (adminRoleModuleRelas != null && adminRoleModuleRelas.Count > 0)
                        {
                            List<string> moduleIDs = adminRoleModuleRelas.Select(p => p.ModuleID).ToList();
                            modules = DBServices.ElincloudBasicInstance.GetMuch<AdminModule>(p => moduleIDs.Contains(p.ID));
                        }
                    }
                }
            }
            else
            {
                modules = DBServices.ElincloudBasicInstance.GetMuch<AdminModule>();
            }

            List<ModuleInfo> rets = new List<ModuleInfo>();
            if (modules != null && modules.Count > 0)
            {
                foreach (var module in modules)
                {
                    rets.Add(new ModuleInfo()
                    {
                        ID = module.ID,
                        PID = module.PID,
                        Name = module.Name,
                        Remark = module.Remark,
                        Action = module.Action,
                        ModulePurpose = ElincloudServer.Common.Utils.EnumUtil.GetEnumValue<ModulePurposes>(module.ModulePurpose),
                        SortIndex = module.SortIndex,
                        Image = module.Image,
                        ResourcesKey = module.ResourcesKey,
                        ThirdUrl = module.ThirdUrl
                    });
                }
            }
            else
            {
                throw new ServiceException("W00000160");
            }
            rets = rets.OrderBy(i => i.SortIndex).ToList();
            return rets;
        }

        /// <summary>
        /// 根据运维角色获取运维模块
        /// </summary>
        /// <param name="modulePurposes"></param>
        /// <param name="roleIDs"></param>
        /// <returns></returns>
        /// <exception cref="ServiceException"></exception>
        public List<ModuleInfo> GetAdminPermittedModulesByAdminRoles(ModulePurposes? modulePurposes, params string[] roleIDs)
        {
            List<ModuleInfo> rets = new List<ModuleInfo>();
            if (roleIDs != null && roleIDs.Length > 0)
            {
                List<AdminRoleModuleRela> roleModuleRelas = DBServices.ElincloudBasicInstance.GetMuch<AdminRoleModuleRela>(p => roleIDs.ToList().Contains(p.RoleID));
                if (roleModuleRelas != null && roleModuleRelas.Count > 0)
                {
                    List<string> moduleIDs = roleModuleRelas.Select(c => c.ModuleID).ToList();
                    Expression<Func<AdminModule, bool>> whereExpression = p => moduleIDs.Contains(p.ID);

                    //modulePurpose为空查所有，不为空按参数来查
                    if (modulePurposes != null)
                    {
                        whereExpression = whereExpression.And(p => p.ModulePurpose == (int)modulePurposes);
                    }

                    List<AdminModule> permittedModules = DBServices.ElincloudBasicInstance.GetMuch<AdminModule>(whereExpression);
                    if (permittedModules != null && permittedModules.Count > 0)
                    {
                        foreach (var module in permittedModules)
                        {
                            ModuleInfo moduleIndo = new ModuleInfo()
                            {
                                ID = module.ID,
                                Name = module.Name,
                                PID = module.PID,
                                Action = module.Action,
                                ModulePurpose = ElincloudServer.Common.Utils.EnumUtil.GetEnumValue<ModulePurposes>(module.ModulePurpose),
                                SortIndex = module.SortIndex,
                                Remark = module.Remark,
                                Image = module.Image,
                                ResourcesKey = module.ResourcesKey,
                                ThirdUrl = module.ThirdUrl
                            };
                            rets.Add(moduleIndo);
                        }
                    }
                    else
                    {
                        throw new ServiceException("W00000160");
                    }
                }
            }
            return rets;
        }

        /// <summary>
        /// 根据运维角色获取医院
        /// </summary>
        /// <param name="roleIDs"></param>
        /// <returns></returns>
        /// <exception cref="ServiceException"></exception>
        public List<string> GetHospitalIDsByAdminRoles(List<string> roleIDs)
        {
            List<string> rets = new List<string>();
            if (roleIDs != null && roleIDs.Count > 0)
            {
                List<AdminRoleHospitalRela> relas = DBServices.ElincloudBasicInstance.GetMuch<AdminRoleHospitalRela>(p => roleIDs.ToList().Contains(p.RoleID));
                if (relas != null && relas.Count > 0)
                {
                    rets = relas.Select(i => i.HospitalID).ToList();
                }
            }
            return rets;
        }

        /// <summary>
        /// 新增运维模块信息
        /// </summary>
        /// <param name="pi"></param>
        /// <param name="moduleInfo"></param>
        /// <returns></returns>
        public void AddAdminModule(PermissionInfo pi, ModuleInfo moduleInfo)
        {
            if (moduleInfo != null)
            {
                AdminModule module = DBServices.ElincloudBasicInstance.GetOne<AdminModule>(c => c.Name == moduleInfo.Name && c.PID == moduleInfo.PID);
                if (module == null)
                {
                    module = new AdminModule()
                    {
                        ID = DBServices.ElincloudBasicInstance.GetID<AdminModule>(),
                        Name = moduleInfo.Name,
                        PID = string.IsNullOrEmpty(moduleInfo.PID) ? ElincloudServer.Common.ApplicationEx.RootID : moduleInfo.PID,
                        Action = moduleInfo.Action,
                        ModulePurpose = (int)moduleInfo.ModulePurpose,
                        SortIndex = moduleInfo.SortIndex,
                        CreateUserID = pi.UserID,
                        CreateDate = DateTime.Now,
                        UpdateUserID = pi.UserID,
                        UpdateDate = DateTime.Now,
                        Remark = moduleInfo.Remark,
                        Image = moduleInfo.Image,
                        ThirdUrl = moduleInfo.ThirdUrl
                    };
                    DBServices.ElincloudBasicInstance.Insert<AdminModule>(module);
                }
                else
                {
                    throw new ServiceException("W00000161");
                }
            }
            else
            {
                throw new ServiceException("W00000162");
            }
        }

        /// <summary>
        /// 更新运维模块信息
        /// </summary>
        /// <param name="moduleInfo"></param>
        public void UpdateAdminModule(PermissionInfo pi, ModuleInfo moduleInfo)
        {
            if (moduleInfo != null)
            {
                AdminModule module = DBServices.ElincloudBasicInstance.GetOne<AdminModule>(m => m.ID == moduleInfo.ID);
                if (module != null)
                {
                    //验证修改后的模块名称不能和已经存在的模块重名
                    AdminModule m = DBServices.ElincloudBasicInstance.GetOne<AdminModule>(p => p.Name == moduleInfo.Name && p.ID != moduleInfo.ID && p.PID == moduleInfo.PID);
                    if (m != null)
                    {
                        throw new ServiceException("W00000161");
                    }

                    module.Name = moduleInfo.Name;
                    module.Action = moduleInfo.Action;
                    module.ModulePurpose = (int)moduleInfo.ModulePurpose;
                    module.SortIndex = moduleInfo.SortIndex;
                    module.UpdateUserID = pi.UserID;
                    module.UpdateDate = DateTime.Now;
                    module.Remark = moduleInfo.Remark;
                    module.Image = moduleInfo.Image;
                    module.ThirdUrl = moduleInfo.ThirdUrl;
                    DBServices.ElincloudBasicInstance.Update<AdminModule>(module, p => p.ID == module.ID,
                        AdminModule._Name,
                        AdminModule._Action,
                        AdminModule._ModulePurpose,
                        AdminModule._SortIndex,
                        AdminModule._UpdateDate,
                        AdminModule._UpdateUserID,
                        AdminModule._Remark,
                        AdminModule._Image,
                        AdminModule._ThirdUrl);
                }
                else
                {
                    throw new ServiceException("W00000163");
                }
            }
            else
            {
                throw new ServiceException("W00000162");
            }
        }

        /// <summary>
        /// 根据运维模块ID删除模块信息
        /// </summary>
        /// <param name="moduleInfo">模块信息</param>
        public void DeleteAdminModule(string moduleID)
        {
            //验证要删除的模块下还有子模块时不允许删除
            int childModuleCount = DBServices.ElincloudBasicInstance.GetCount<AdminModule>(p => p.PID == moduleID);
            if (childModuleCount > 0)
            {
                throw new ServiceException("W00000164");
            }
            //删除当前模块
            DBServices.ElincloudBasicInstance.Delete<AdminModule>(p => p.ID == moduleID);
        }

        /// <summary>
        /// 根据运维角色信息维护运维权限信息
        /// </summary>
        /// <param name="roleID"></param>
        public void UpdateAdminPermisionByAdminRole(string roleID, List<string> moduleIDs, List<string> hospitalIDs)
        {
            if (!string.IsNullOrEmpty(roleID))
            {
                if (moduleIDs == null || moduleIDs.Count == 0)
                {
                    throw new ServiceException("W00000167");
                }

                if (hospitalIDs == null || hospitalIDs.Count == 0)
                {
                    throw new ServiceException("W00000452");
                }

                Transaction tran = DBServices.ElincloudBasicInstance.GetTransaction();

                tran.Delete<AdminRoleModuleRela>(p => p.RoleID == roleID);
                List<AdminRoleModuleRela> modules = moduleIDs.Select(i =>
                {
                    return new AdminRoleModuleRela()
                    {
                        ModuleID = i,
                        RoleID = roleID,
                    };
                }).ToList();
                tran.Insert<AdminRoleModuleRela>(modules.ToArray());

                tran.Delete<AdminRoleHospitalRela>(p => p.RoleID == roleID);
                List<AdminRoleHospitalRela> hospitals = hospitalIDs.Select(i =>
                {
                    return new AdminRoleHospitalRela()
                    {
                        HospitalID = i,
                        RoleID = roleID,
                    };
                }).ToList();
                tran.Insert<AdminRoleHospitalRela>(hospitals.ToArray());

                if (!tran.Commit())
                {
                    throw new ServiceException("W00000166");
                }
            }
            else
            {
                throw new ServiceException("W00000168");
            }
        }
        #endregion

        

        #region 检查类型

        /// <summary>
        /// 获取检查类型列表 
        /// </summary>
        /// <returns></returns>
        /// <exception cref="ServiceException"></exception>
        public List<CheckInfo> GetCheckInfos()
        {
            return CommonBL.GetCheckInfos();
        }

        #endregion        
    }
}
