using ElincloudAISliceServer.Entity;
using ElincloudServer.BLServices;
using ElincloudServer.Common;
using ElincloudServer.CommonBLService;
using ElincloudServer.Entity;
using ElincloudServer.Entity.IM.Enums;
using Microsoft.AspNetCore.Mvc;
using SixLabors.ImageSharp;
using System.Reflection.Emit;

namespace ElincloudServer.Controllers
{
    public class SystemBLServiceController : BaseController
    {
        #region 独立服务相关

        /// <summary>
        /// 获取服务列表信息
        /// </summary>
        /// <returns></returns>
        public JsonResult GetServiceInfos()
        {
            return Excute<List<ServiceInfo>>(new Func<PermissionInfo, List<ServiceInfo>>((PermissionInfo pi) =>
            {
                return CommonBL.GetServiceInfos();
            }), "E00000414");
        }

        private async Task StartServicesWithThrottleAsync(List<ServiceInfo> serviceList, ForwardingSystemCommandTypes CommandTypes, int maxConcurrency = 5, int delayMs = 500)
        {
            // 使用 SemaphoreSlim 来控制并发任务的最大数量，防止系统资源过载。
            using var semaphore = new SemaphoreSlim(maxConcurrency);
            var tasks = new List<Task>();

            foreach (var service in serviceList)
            {
                await semaphore.WaitAsync(); // 等待获取信号量（并发名额）

                tasks.Add(Task.Run(() =>
                {
                    try
                    {
                        ElincloudServer.CommonBLService.CommonBL.RunServiceByCommand(
                            service.ServiceENName,
                            CommandTypes,
                             new List<ServiceInfo> { service }, "");
                    }
                    catch
                    {
                    }
                    finally
                    {
                        semaphore.Release();// 释放信号量，允许下一个任务开始
                    }
                }));
                await Task.Delay(delayMs); // 控制每个任务启动的节奏，避免启动太快导致CPU爆炸
            }
            await Task.WhenAll(tasks);

        }

        public JsonResult RunServiceByCommand(List<ServiceInfo> serviceList, ControllerParameter<ForwardingSystemCommandTypes> commandTypes)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                ElincloudServer.Common.Utils.ThreadUtil.StartThread(new
              Action(async () =>
              {
                  if (serviceList != null && serviceList.Count > 0)
                  {
                      serviceList = serviceList.OrderByDescending(p => p.ServiceRunLogPort).ToList();//远程仓储服务与数据存储服务如需在同一台机器运行时，需要先运行远程仓储服务
                      await StartServicesWithThrottleAsync(serviceList, commandTypes.Value);
                  }
              }));
            }), "E00000415");
        }
        #endregion

        #region 锁屏心跳
        public JsonResult Heartbeat()
        {
            return Excute(new Action(() => { }), "");
        }

        #endregion

        #region 检查类型
        /// <summary>
        /// 获取检查类型列表
        /// </summary>
        /// <returns></returns>

        public JsonResult GetCheckInfos()
        {
            return Excute<List<CheckInfo>>(new Func<PermissionInfo, List<CheckInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetCheckInfos();
            }), "E00000328");
        }

        #endregion

        #region 地址管理

        public JsonResult GetHospitalAddressInfos(string hospitalGroupID)
        {
            return Excute<List<AddressInfo>>(new Func<PermissionInfo, List<AddressInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetHospitalAddressInfos(hospitalGroupID);
            }), "E00000299");
        }

        public JsonResult AddAddressInfo(string hospitalGroupID, AddressInfo addressInfo)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.AddAddressInfo(pi, hospitalGroupID, addressInfo);
            }), "E00000300");
        }

        public JsonResult UpdateAddressInfo(string hospitalGroupID, AddressInfo addressInfo)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.UpdateAddressInfo(pi, hospitalGroupID, addressInfo);
            }), "E00000301");
        }

        public JsonResult DeleteAddressInfo(string hospitalGroupID, string addressID)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.DeleteAddressInfo(hospitalGroupID, addressID);
            }), "E00000302");
        }

        public JsonResult GetAddressInfosByPID(string hospitalGroupID, string searchStr, string parentID = ElincloudServer.Common.ApplicationEx.RootID)
        {
            return Excute<List<AddressInfo>>(new Func<PermissionInfo, List<AddressInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetAddressInfosByPID(hospitalGroupID, searchStr, parentID);
            }), "E00000299");
        }
        #endregion


        #region 医院管理

        public JsonResult GetHospitals()
        {
            return Excute<List<HospitalInfo>>(new Func<PermissionInfo, List<HospitalInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetHospitals();
            }), "E00000275");
        }

        public JsonResult GetHospitalInfosByPage(ControllerParameter<int> pageIndex, ControllerParameter<int> pageSize)
        {
            return Excute<Page<HospitalInfo>>(new Func<PermissionInfo, Page<HospitalInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetHospitalInfosByPage(pageIndex.Value, pageSize.Value);
            }), "E00000583");
        }
        /// <summary>
        /// 根据医院分组ID获取医院列表
        /// </summary>
        /// <param name="hospitalGroupID"></param>
        /// <returns></returns>
        /// <remarks>
        /// sle 2025年5月28日15:56:27
        /// </remarks>
        public JsonResult GetHospitalsByHospitalGroupID(string hospitalGroupID)
        {
            return Excute<List<HospitalInfo>>(new Func<PermissionInfo, List<HospitalInfo>>((PermissionInfo pi) =>
            {
                List<HospitalInfo> list = BLService.SystemBLService.GetHospitals();
                list = list.FindAll(i => i.HospitalGroupID == hospitalGroupID);
                return list;
            }), "E00000276");
        }

        public JsonResult AddHospital(HospitalInfo hospitalInfo)
        {
            return Excute(new Func<PermissionInfo, string>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.AddHospital(hospitalInfo);
            }), "E00000277");
        }

        public JsonResult UpdateHospital(HospitalInfo hospitalInfo)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.UpdateHospital(hospitalInfo);
            }), "E00000278");
        }

        #endregion

        #region 医院授权

        public JsonResult GetHospitalRegisterInfos()
        {
            return Excute<List<HospitalRegistInfo>>(new Func<PermissionInfo, List<HospitalRegistInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetHospitalRegisterInfos();
            }), "E00000398");
        }

        public JsonResult AddHospitalRegisterInfo(HospitalRegistInfo info)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.AddHospitalRegisterInfo(info);
            }), "E00000399");
        }

        public JsonResult UpdateHospitalRegisterInfo(HospitalRegistInfo info)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.UpdateHospitalRegisterInfo(info);
            }), "E00000400");
        }

        public JsonResult DeleteHospitalRegisterInfoByHospitalID(string hospitalID)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.DeleteHospitalRegisterInfoByHospitalID(hospitalID);
            }), "E00000401");
        }

        #endregion

        #region 日志查询

        #region SQL日志查询

        [DisableAuditAttribute]
        public JsonResult GetSQLLogInfosByPage(ControllerParameter<DateTime> startDate, ControllerParameter<DateTime> endDate, string searchString, string hospitalNumber, ControllerParameter<int> pageIndex, ControllerParameter<int> pageSize)
        {
            return Excute<Page<SQLLogInfo>>(new Func<PermissionInfo, Page<SQLLogInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetSQLLogInfosByPage(startDate.Value, endDate.Value, searchString, hospitalNumber, pageIndex.Value, pageSize.Value);
            }), "E00000410");
        }
        #endregion

        #region 审计日志

        [DisableAuditAttribute]
        public JsonResult GetAuditLogInfosByPage(ControllerParameter<DateTime> startDate, ControllerParameter<DateTime> endDate, string searchString, string hospitalNumber, ControllerParameter<int> pageIndex, ControllerParameter<int> pageSize)
        {
            return Excute<Page<AuditLogInfo>>(new Func<PermissionInfo, Page<AuditLogInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetAuditLogInfosByPage(startDate.Value, endDate.Value, searchString, hospitalNumber, pageIndex.Value, pageSize.Value);
            }), "E00000411");
        }
        #endregion

        #region 错误日志

        [DisableAuditAttribute]
        public JsonResult GetErrorLogInfos(ControllerParameter<DateTime> startDate, ControllerParameter<LogTypes> logType, string hospitalNumber)
        {
            return Excute<List<ErrorLogInfo>>(new Func<PermissionInfo, List<ErrorLogInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetErrorLogInfos(startDate.Value, logType.Value, hospitalNumber);
            }), "E00000412");
        }

        [DisableAuditAttribute]
        public JsonResult GetErrorLogContent(ControllerParameter<DateTime> startDate, ControllerParameter<LogTypes> logType, string logFileName, string hospitalNumber)
        {
            return Excute<string>(new Func<PermissionInfo, string>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetErrorLogContent(startDate.Value, logType.Value, logFileName, hospitalNumber);
            }), "E00000413");
        }

        #endregion

        #region 运行日志

        [DisableAuditAttribute]
        /// <summary>
        /// 获取运行日志的列表
        /// </summary>
        /// <param name="startDate">开始日期</param>
        /// <param name="logType">日志类型</param>
        /// <param name="hospitalNumber">医院code</param>
        /// <returns>{"Result":"List<ErrorLogInfo>","Success":true,"Message":""}</returns>
        public JsonResult GetRunLogInfos(ControllerParameter<DateTime> startDate, ControllerParameter<LogTypes> logType, string hospitalNumber)
        {
            return Excute<List<ErrorLogInfo>>(new Func<PermissionInfo, List<ErrorLogInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetRunLogInfos(startDate.Value, logType.Value, hospitalNumber);
            }), "E00000412");
        }

        [DisableAuditAttribute]
        /// <summary>
        /// 获取对应的日志内容
        /// </summary>
        /// <param name="startDate">开始日期</param>
        /// <param name="logType">日志类型</param>
        /// <param name="logFileName">文件名称</param>
        /// <param name="hospitalNumber">医院code</param>
        /// <returns>{"Result":"日志内容","Success":true,"Message":""}</returns>
        public JsonResult GetRunLogContent(ControllerParameter<DateTime> startDate, ControllerParameter<LogTypes> logType, string logFileName, string hospitalNumber)
        {
            return Excute<string>(new Func<PermissionInfo, string>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetRunLogContent(startDate.Value, logType.Value, logFileName, hospitalNumber);
            }), "E00000413");
        }

        #endregion

        #endregion



        #region 用户管理
        public JsonResult UpdatePassword(string userID, string oldPwd, string newPwd)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.UpdatePassword(pi, userID, oldPwd, newPwd);
            }), "E00000253");
        }

        public JsonResult GetHospitalUsers(string hospitalID, string hospitalGroupID)
        {
            return Excute<List<UserInfo>>(new Func<PermissionInfo, List<UserInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetHospitalUsers(hospitalID, hospitalGroupID);
            }), "E00000254");
        }

        public JsonResult GetHospitalUsersByPage(string hospitalID, string hospitalGroupID, string searchStr, ControllerParameter<int> pageIndex, ControllerParameter<int> pageSize)
        {
            return Excute<Page<DoctorInfo>>(new Func<PermissionInfo, Page<DoctorInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetHospitalUsersByPage(hospitalID, hospitalGroupID, searchStr, pageIndex.Value, pageSize.Value);
            }), "E00000255");
        }

        public JsonResult GetUserByID(string hospitalGroupID, string userID)
        {
            return Excute<UserInfo>(new Func<PermissionInfo, UserInfo>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetUserByID(hospitalGroupID, userID);
            }), "E00000258");
        }

        public JsonResult GetUserByName(string hospitalGroupID, string userName)
        {
            return Excute<UserInfo>(new Func<PermissionInfo, UserInfo>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetUserByName(hospitalGroupID, userName);
            }), "E00000259");
        }

        public JsonResult GetDetailUserInfoByID(string hospitalGroupID, string userID)
        {
            return Excute<DetailUserInfo>(new Func<PermissionInfo, DetailUserInfo>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetDetailUserInfoByID(hospitalGroupID, userID);
            }), "E00000260");
        }

        public JsonResult AddUser(string hospitalID, string hospitalGroupID, string hospitalGroupCode, UserInfo user)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.AddUser(pi, hospitalID, hospitalGroupID, hospitalGroupCode, user);
            }), "E00000261");
        }

        public JsonResult UpdateUser(string hospitalGroupID, string hospitalGroupCode, UserInfo user)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.UpdateUser(pi, hospitalGroupID, hospitalGroupCode, user);
            }), "E00000262");
        }

        public JsonResult UpdateUserStatus(string userID, ControllerParameter<SysStatus> state)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.UpdateUserStatus(pi, userID, state.Value);
            }), "E00000263");
        }

        public JsonResult GetUsersByRole(string hospitalGroupID, string roleID)
        {
            return Excute<List<UserInfo>>(new Func<PermissionInfo, List<UserInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetUsersByRole(hospitalGroupID, roleID);
            }), "E00000264");
        }

        public JsonResult GetUsersByRoleByPage(string hospitalGroupID, string roleID, ControllerParameter<int> pageIndex, ControllerParameter<int> pageSize)
        {
            return Excute<Page<UserInfo>>(new Func<PermissionInfo, Page<UserInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetUsersByRoleByPage(hospitalGroupID, roleID, pageIndex.Value, pageSize.Value);
            }), "E00000265");
        }
        public JsonResult UpdateUserRoleRelas(string hospitalGroupID, string userID, List<string> roleIDs)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.UpdateUserRoleRelas(hospitalGroupID, userID, roleIDs);
            }), "E00000266");
        }

        public JsonResult DeleteUserRole(string hospitalGroupID, string userID, string roleID)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.DeleteUserRole(hospitalGroupID, userID, roleID);
            }), "E00000267");
        }

        public JsonResult GetUsersByOrgnization(string hospitalGroupID, string orgID)
        {
            return Excute<List<UserInfo>>(new Func<PermissionInfo, List<UserInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetUsersByOrgnization(hospitalGroupID, orgID);
            }), "E00000268");
        }

        public JsonResult GetUsersByOrgnizationByPage(string hospitalGroupID, string searchStr, string orgID, ControllerParameter<int> pageIndex, ControllerParameter<int> pageSize)
        {
            return Excute<Page<UserInfo>>(new Func<PermissionInfo, Page<UserInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetUsersByOrgnizationByPage(hospitalGroupID, searchStr, orgID, pageIndex.Value, pageSize.Value);
            }), "E00000268");
        }
        public JsonResult GetUsersWithoutOrganization(string hospitalID, string hospitalGroupID)
        {
            return Excute<List<UserInfo>>(new Func<PermissionInfo, List<UserInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetUsersWithoutOrganization(hospitalID, hospitalGroupID);
            }), "E00000270");
        }

        public JsonResult UpdateUserOrganizationRela(string hospitalGroupID, string userID, string orgId)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.UpdateUserOrganizationRela(hospitalGroupID, userID, orgId);
            }), "E00000271");
        }

        public JsonResult GetSimpleUserInfosByPage(string hospitalID, string hospitalGroupID, string searchStr, ControllerParameter<int> pageIndex, ControllerParameter<int> pageSize)
        {
            return Excute<Page<SimpleUserInfo>>(new Func<PermissionInfo, Page<SimpleUserInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetSimpleUserInfosByPage(hospitalID, hospitalGroupID, searchStr, pageIndex.Value, pageSize.Value);
            }), "E00000272");
        }

        public JsonResult GetSimpleUserInfosByRoleByPage(string hospitalGroupID, string roleID, string searchStr, ControllerParameter<int> pageIndex, ControllerParameter<int> pageSize)
        {
            return Excute<Page<SimpleUserInfo>>(new Func<PermissionInfo, Page<SimpleUserInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetSimpleUserInfosByRoleByPage(hospitalGroupID, roleID, searchStr, pageIndex.Value, pageSize.Value);
            }), "E00000265");
        }

        public JsonResult GetSimpleUserInfos(string hospitalID, string hospitalGroupID)
        {
            return Excute<List<SimpleUserInfo>>(new Func<PermissionInfo, List<SimpleUserInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetSimpleUserInfos(hospitalID, hospitalGroupID);
            }), "E00000272");
        }

        #region 注册相关

        public JsonResult RegisterUserInfo(string hospitalID, string hospitalGroupID, UserInfo user, string roleID, string orgID, string operationUserID, string registerURL)
        {
            return Excute((() =>
            {
                BLService.SystemBLService.RegisterUserInfo(hospitalID, hospitalGroupID, user, roleID, orgID, operationUserID, registerURL);
            }), "W00000418");
        }

        public JsonResult SendVerificationCode(string hospitalID, string telePhone)
        {
            return Excute((() =>
            {
                BLService.SystemBLService.SendVerificationCode(hospitalID, telePhone);
            }), "E00000190");
        }

        public JsonResult CheckVerificationCode(string telePhone, string verificationCode)
        {
            return Excute((() =>
            {
                BLService.SystemBLService.CheckVerificationCode(telePhone, verificationCode);
            }), "E00000469");
        }

        #endregion

        #endregion

        #region 角色管理
        public JsonResult GetHospitalRoles(string hospitalID, string hospitalGroupID, string searchStr)
        {
            return Excute<List<RoleInfo>>(new Func<PermissionInfo, List<RoleInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetHospitalRoles(hospitalID, hospitalGroupID, searchStr);
            }), "E00000287");
        }

        public JsonResult GetHospitalRolesByPage(string hospitalID, string hospitalGroupID, string searchStr, ControllerParameter<int> pageIndex, ControllerParameter<int> pageSize)
        {
            return Excute<Page<RoleInfo>>(new Func<PermissionInfo, Page<RoleInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetHospitalRolesByPage(hospitalID, hospitalGroupID, searchStr, pageIndex.Value, pageSize.Value);
            }), "E00000287");
        }

        public JsonResult GetSimpleUsersWithoutAnyRole(string hospitalID, string hospitalGroupID)
        {
            return Excute<List<SimpleUserInfo>>(new Func<PermissionInfo, List<SimpleUserInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetSimpleUsersWithoutAnyRole(hospitalID, hospitalGroupID);
            }), "E00000287");
        }

        public JsonResult AddRole(string hospitalID, string hospitalGroupID, RoleInfo role)
        {
            return Excute<string>(new Func<PermissionInfo, string>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.AddRole(pi, hospitalID, hospitalGroupID, role);
            }), "E00000290");
        }

        public JsonResult UpdateRole(string hospitalID, string hospitalGroupID, RoleInfo role)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.UpdateRole(pi, hospitalID, hospitalGroupID, role);
            }), "E00000291");
        }

        public JsonResult DeleteRole(string hospitalGroupID, string roleID)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.DeleteRole(pi, hospitalGroupID, roleID);
            }), "E00000292");
        }

        public JsonResult AddDoctorRoleRelas(string hospitalGroupID, string roleID, List<string> userIds)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.AddDoctorRoleRelas(pi, hospitalGroupID, roleID, userIds);
            }), "E00000293");
        }

        #endregion

        #region 机构管理

        /// <summary>
        /// 获取医院分组下的组织机构
        /// </summary>
        /// <returns></returns>
        public JsonResult GetOrganizationInfoByHospitalGroupID(string hospitalGroupID)
        {
            return Excute<List<OrganizationInfo>>(new Func<PermissionInfo, List<OrganizationInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetOrganizationInfoByHospitalGroupID(hospitalGroupID);
            }), "E00000279");
        }

        public JsonResult GetHospitalOrganizations(string hospitalID, string hospitalGroupID, string searchStr)
        {
            return Excute<List<OrganizationInfo>>(new Func<PermissionInfo, List<OrganizationInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetHospitalOrganizations(hospitalID, hospitalGroupID, searchStr);
            }), "E00000280");
        }

        public JsonResult GetOrganizationInfoByUserID(string hospitalGroupID, string userID)
        {
            return Excute<OrganizationInfo>(new Func<PermissionInfo, OrganizationInfo>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetOrganizationInfoByUserID(hospitalGroupID, userID);
            }), "E00000281");
        }

        public JsonResult GetSimpleUsersWithoutAnyOrganization(string hospitalID, string hospitalGroupID)
        {
            return Excute<List<SimpleUserInfo>>(new Func<PermissionInfo, List<SimpleUserInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetSimpleUsersWithoutAnyOrganization(hospitalID, hospitalGroupID);
            }), "E00000272");
        }

        public JsonResult AddOrganization(string hospitalID, string hospitalGroupID, OrganizationInfo organization)
        {
            return Excute<string>(new Func<PermissionInfo, string>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.AddOrganization(pi, hospitalID, hospitalGroupID, organization);
            }), "E00000283");
        }

        public JsonResult UpdateOrganization(string hospitalID, string hospitalGroupID, OrganizationInfo organization)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.UpdateOrganization(pi, hospitalID, hospitalGroupID, organization);
            }), "E00000284");
        }

        public JsonResult DeleteOrganization(string hospitalGroupID, string organizationID)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.DeleteOrganization(pi, hospitalGroupID, organizationID);
            }), "E00000285");
        }

        public JsonResult UpdateUsersOrganizationRelas(string hospitalGroupID, string organizationID, List<string> userIDs)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.UpdateUsersOrganizationRelas(hospitalGroupID, organizationID, userIDs);
            }), "E00000286");
        }

        #endregion

        #region 模块管理

        public JsonResult GetModules(string hospitalGroupID)
        {
            return Excute<List<ModuleInfo>>(new Func<PermissionInfo, List<ModuleInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetModules(hospitalGroupID);
            }), "E00000245");
        }

        /// <summary>
        /// 根据角色获取模块列表
        /// </summary>
        /// <param name="hospitalGroupID"></param>
        /// <param name="roleID"></param>
        /// <returns></returns>
        public JsonResult GetPermittedModulesByRole(string hospitalGroupID, string roleID)
        {
            return Excute<List<ModuleInfo>>(new Func<PermissionInfo, List<ModuleInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetPermittedModulesByRoles(hospitalGroupID, null, roleID);
            }), "E00000250");
        }

        /// <summary>
        /// 根据角色列表获取模块列表
        /// </summary>
        /// <param name="hospitalGroupID"></param>
        /// <param name="roleIDs"></param>
        /// <param name="modulePurpose"></param>
        /// <returns></returns>
        /// <exception cref="ServiceException"></exception>
        public JsonResult GetPermittedModulesByRoles(string hospitalGroupID, List<string> roleIDs, ControllerParameter<ModulePurposes> modulePurpose)
        {
            return Excute<List<ModuleInfo>>(new Func<PermissionInfo, List<ModuleInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetPermittedModulesByRoles(hospitalGroupID, modulePurpose.Value, roleIDs.ToArray());
            }), "E00000251");
        }

        public JsonResult AddModule(string hospitalGroupID, ModuleInfo module)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.AddModule(pi, hospitalGroupID, module);
            }), "E00000247");
        }

        public JsonResult UpdateModule(string hospitalGroupID, ModuleInfo module)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.UpdateModule(pi, hospitalGroupID, module);
            }), "E00000248");
        }

        public JsonResult DeleteModule(string hospitalGroupID, string moduleID)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.DeleteModule(hospitalGroupID, moduleID);
            }), "E00000249");
        }

        /// <summary>
        /// 根据角色信息维护权限信息
        /// </summary>
        /// <param name="hospitalGroupID"></param>
        /// <param name="roleID"></param>
        /// <param name="permittedModuleIDs"></param>
        /// <returns></returns>
        public JsonResult UpdatePermisionByRole(string hospitalGroupID, string roleID, List<string> permittedModuleIDs)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.UpdatePermisionByRole(hospitalGroupID, roleID, permittedModuleIDs);
            }), "E00000252");
        }

        #endregion



        #region 运维用户管理
        /// <summary>
        /// 根据用户ID获取运维用户
        /// </summary>
        /// <param name="userID"></param>
        /// <returns></returns>
        public JsonResult GetAdminUserByID(string userID)
        {
            return Excute<UserInfo>(new Func<PermissionInfo, UserInfo>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetAdminUserByID(userID);
            }), "E00000522");
        }

        /// <summary>
        /// 获取所有运维用户
        /// </summary>
        /// <returns></returns>
        public JsonResult GetAdminUsers()
        {
            return Excute<List<UserInfo>>(new Func<PermissionInfo, List<UserInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetAdminUsers();
            }), "E00000523");
        }

        /// <summary>
        /// 分页获取运维用户列表
        /// </summary>
        /// <param name="pageIndex"></param>
        /// <param name="pageSize"></param>
        /// <returns></returns>
        public JsonResult GetAdminUsersByPage(ControllerParameter<int> pageIndex, ControllerParameter<int> pageSize)
        {
            return Excute<Page<UserInfo>>(new Func<PermissionInfo, Page<UserInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetAdminUsersByPage(pageIndex.Value, pageSize.Value);
            }), "E00000524");
        }

        /// <summary>
        /// 根据运维角色ID获取运维用户列表
        /// </summary>
        /// <param name="roleID"></param>
        /// <returns></returns>
        public JsonResult GetAdminUsersByRole(string roleID)
        {
            return Excute<List<UserInfo>>(new Func<PermissionInfo, List<UserInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetAdminUsersByRole(roleID);
            }), "E00000525");
        }

        /// <summary>
        /// 根据运维用户角色ID分页获取运维用户列表
        /// </summary>
        /// <param name="roleID"></param>
        /// <param name="pageIndex"></param>
        /// <param name="pageSize"></param>
        /// <returns></returns>
        public JsonResult GetAdminUsersByRoleByPage(string roleID, ControllerParameter<int> pageIndex, ControllerParameter<int> pageSize)
        {
            return Excute<Page<UserInfo>>(new Func<PermissionInfo, Page<UserInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetAdminUsersByRoleByPage(roleID, pageIndex.Value, pageSize.Value);
            }), "E00000526");
        }

        /// <summary>
        /// 设置管理员用户启用、停用状态
        /// </summary>
        /// <param name="userID"></param>
        /// <param name="state"></param>
        /// <returns></returns>
        public JsonResult UpdateAdminUserStatus(string userID, ControllerParameter<SysStatus> state)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.UpdateAdminUserStatus(pi, userID, state.Value);
            }), "E00000527");
        }

        /// <summary>
        /// 新增运维用户
        /// </summary>
        /// <param name="user"></param>
        /// <returns>ID</returns>
        public JsonResult AddAdminUser(UserInfo user)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.AddAdminUser(pi, user);
            }), "E00000528");
        }

        /// <summary>
        /// 更新运维用户
        /// </summary>
        /// <param name="user"></param>
        public JsonResult UpdateAdminUser(UserInfo user)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.UpdateAdminUser(pi, user);
            }), "E00000529");
        }

        /// <summary>
        /// 修改运维用户的角色
        /// </summary>
        /// <param name="userID"></param>
        /// <param name="roleIDs"></param>
        /// <returns></returns>
        public JsonResult UpdateAdminUserRoleRelas(string userID, List<string> roleIDs)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.UpdateAdminUserRoleRelas(userID, roleIDs);
            }), "E00000530");
        }
        #endregion

        #region 运维角色管理
        /// <summary>
        /// 获取所有的运维角色
        /// </summary>
        /// <param name="searchStr"></param>
        /// <returns></returns>
        public JsonResult GetAdminRoles(string searchStr)
        {
            return Excute<List<RoleInfo>>(new Func<PermissionInfo, List<RoleInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetAdminRoles(searchStr);
            }), "E00000531");
        }

        /// <summary>
        /// 分页获取运维角色
        /// </summary>
        /// <param name="searchStr"></param>
        /// <param name="pageIndex"></param>
        /// <param name="pageSize"></param>
        /// <returns></returns>
        public JsonResult GetAdminRolesByPage(string searchStr, ControllerParameter<int> pageIndex, ControllerParameter<int> pageSize)
        {
            return Excute<Page<RoleInfo>>(new Func<PermissionInfo, Page<RoleInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetAdminRolesByPage(searchStr, pageIndex.Value, pageSize.Value);
            }), "E00000532");
        }

        /// <summary>
        /// 获取不在运维角色下的运维用户
        /// </summary>
        /// <returns></returns>
        public JsonResult GetSimpleAdminUsersWithoutAnyRole()
        {
            return Excute<List<SimpleUserInfo>>(new Func<PermissionInfo, List<SimpleUserInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetSimpleAdminUsersWithoutAnyRole();
            }), "E00000533");
        }

        /// <summary>
        /// 添加运维角色
        /// </summary>
        /// <param name="role"></param>
        /// <returns></returns>
        public JsonResult AddAdminRole(RoleInfo role)
        {
            return Excute<string>(new Func<PermissionInfo, string>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.AddAdminRole(pi, role);
            }), "E00000534");
        }

        /// <summary>
        /// 修改运维角色
        /// </summary>
        /// <param name="role"></param>
        /// <returns></returns>
        public JsonResult UpdateAdminRole(RoleInfo role)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.UpdateAdminRole(pi, role);
            }), "E00000535");
        }

        /// <summary>
        /// 删除运维角色
        /// </summary>
        /// <param name="roleID"></param>
        /// <returns></returns>
        public JsonResult DeleteAdminRole(string roleID)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.DeleteAdminRole(pi, roleID);
            }), "E00000536");
        }

        /// <summary>
        /// 给运维角色下添加运维用户
        /// </summary>
        /// <param name="roleID"></param>
        /// <param name="userIds"></param>
        /// <returns></returns>
        public JsonResult AddAdminRoleRelas(string roleID, List<string> userIds)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.AddAdminRoleRelas(pi, roleID, userIds);
            }), "E00000537");
        }

        /// <summary>
        /// 删除运维角色下的用戶
        /// </summary>
        /// <param name="userID"></param>
        /// <param name="roleID"></param>
        /// <returns></returns>
        public JsonResult DeleteAdminRoleUser(string userID, string roleID)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.DeleteAdminRoleUser(userID, roleID);
            }), "E00000538");
        }

        #endregion

        #region 运维模块管理
        /// <summary>
        /// 根据用户名获取运维模块
        /// </summary>
        /// <param name="loginName"></param>
        /// <returns></returns>
        public JsonResult GetPermittedAdminModulesByLoginName(string loginName)
        {
            return Excute<List<ModuleInfo>>(new Func<PermissionInfo, List<ModuleInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetPermittedAdminModulesByLoginName(loginName);
            }), "E00000515");
        }

        /// <summary>
        /// 根据运维角色获取运维模块列表
        /// </summary>
        /// <param name="roleID"></param>
        /// <returns></returns>
        public JsonResult GetAdminPermittedModulesByAdminRole(string roleID)
        {
            return Excute<List<ModuleInfo>>(new Func<PermissionInfo, List<ModuleInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetAdminPermittedModulesByAdminRoles(null, roleID);
            }), "E00000516");
        }

        /// <summary>
        /// 根据运维角色列表获取运维模块列表
        /// </summary>
        /// <param name="roleIDs"></param>
        /// <param name="modulePurpose"></param>
        /// <returns></returns>
        /// <exception cref="ServiceException"></exception>
        public JsonResult GetAdminPermittedModulesByAdminRoles(List<string> roleIDs, ControllerParameter<ModulePurposes> modulePurpose)
        {
            return Excute<List<ModuleInfo>>(new Func<PermissionInfo, List<ModuleInfo>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetAdminPermittedModulesByAdminRoles(modulePurpose.Value, roleIDs.ToArray());
            }), "E00000517");
        }

        /// <summary>
        /// 根据运维角色列表获取运维模块列表
        /// </summary>
        /// <param name="roleIDs"></param>
        /// <returns></returns>
        /// <exception cref="ServiceException"></exception>
        public JsonResult GetHospitalIDsByAdminRoles(List<string> roleIDs)
        {
            return Excute<List<string>>(new Func<PermissionInfo, List<string>>((PermissionInfo pi) =>
            {
                return BLService.SystemBLService.GetHospitalIDsByAdminRoles(roleIDs);
            }), "E00000539");
        }

        /// <summary>
        /// 添加运维模块
        /// </summary>
        /// <param name="module"></param>
        /// <returns></returns>
        public JsonResult AddAdminModule(ModuleInfo module)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.AddAdminModule(pi, module);
            }), "E00000518");
        }

        /// <summary>
        /// 修改运维模块
        /// </summary>
        /// <param name="module"></param>
        /// <returns></returns>
        public JsonResult UpdateAdminModule(ModuleInfo module)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.UpdateAdminModule(pi, module);
            }), "E00000519");
        }

        /// <summary>
        /// 删除运维模块
        /// </summary>
        /// <param name="moduleID"></param>
        /// <returns></returns>
        public JsonResult DeleteAdminModule(string moduleID)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.DeleteAdminModule(moduleID);
            }), "E00000520");
        }

        /// <summary>
        /// 根据运维角色信息维护权限信息
        /// </summary>
        /// <param name="roleID"></param>
        /// <param name="permittedModuleIDs"></param>
        /// <returns></returns>
        public JsonResult UpdateAdminPermisionByAdminRole(string roleID, List<string> permittedModuleIDs, List<string> hospitalIDs)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.SystemBLService.UpdateAdminPermisionByAdminRole(roleID, permittedModuleIDs, hospitalIDs);
            }), "E00000521");
        }

        #endregion




    }
}
