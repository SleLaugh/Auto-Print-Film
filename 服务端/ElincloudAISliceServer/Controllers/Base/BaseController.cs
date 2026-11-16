using Microsoft.AspNetCore.Mvc;
using ElincloudServer.Entity;
using ElincloudServer.Common;
using ElincloudServer.BLServices;

namespace ElincloudServer.Controllers
{

    [Audit]
    public class BaseController : Controller
    {
        #region Excute执行工具

        /// <summary>
        /// 执行带有权限验证和错误处理的函数。
        /// </summary>
        /// <typeparam name="TResult">函数返回结果的类型。</typeparam>
        /// <param name="action">要执行的函数。</param>
        /// <param name="afterExcute">操作完成后执行的委托（可选）。</param>
        /// <param name="safeMode">安全模式标志，指示是否在安全模式下执行（可选）。</param>
        /// <returns>包含执行结果的 JsonResult 对象。</returns>
        public JsonResult Excute<TResult>(Func<PermissionInfo, TResult> action, string exceptionMsg)
        {
            ClientResult serviceRet = null;
            try
            {
                if (IMCenterUtil.IMCenter.BeforeExcute != null)
                {
                    PermissionInfo pi = IMCenterUtil.IMCenter.BeforeExcute();
                    if (pi.IsPermitted)
                    {
                        TResult ret = action(pi);
                        serviceRet = new ClientResult() { Success = true, Result = ret };

                    }
                    else
                    {
                        serviceRet = new ClientResult() { Success = false, Result = default(TResult), Message = "无权限调用该方法！" };
                    }
                }
                else
                {
                    serviceRet = new ClientResult() { Success = false, Result = default(TResult), Message = "未初始化接口权限验证方法！" };
                }
            }
            catch (ServiceException ex)
            {
                serviceRet = new ClientResult() { Success = false, Message = ex.Message };
                if (ex.InnerException != null)
                {
                    ElincloudServer.Common.Utils.LogUtil.RecordErrorLog(GetExceptionMsg(ex.Message), ex.InnerException);
                }
                else
                {
                    ElincloudServer.Common.Utils.LogUtil.RecordErrorLog(GetExceptionMsg(ex.Message));
                }
            }
            catch (Exception ex)
            {
                serviceRet = new ClientResult() { Success = false, Message = ex.Message };
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog(GetExceptionMsg(ex.Message), ex);
            }

            if (!serviceRet.Success)
            {
                serviceRet.Result = 200;
            }
            return new JsonResult(serviceRet);
        }
        /// <summary>
        /// 执行带有权限验证和错误处理的动作。
        /// </summary>
        /// <typeparam name="TResult">操作结果的类型。</typeparam>
        /// <param name="action">要执行的动作。</param>
        /// <param name="afterExcute">操作完成后执行的委托（可选）。</param>
        /// <param name="safeMode">安全模式标志，指示是否在安全模式下执行（可选）。</param>
        /// <returns>包含执行结果的 JsonResult 对象。</returns>
        public JsonResult Excute(Action<PermissionInfo> action, string exceptionMsg)
        {
            ClientResult serviceRet = null;
            try
            {
                if (IMCenterUtil.IMCenter.BeforeExcute != null)
                {
                    PermissionInfo pi = IMCenterUtil.IMCenter.BeforeExcute();
                    if (pi.IsPermitted)
                    {
                        action(pi);
                        serviceRet = new ClientResult() { Success = true };
                    }
                    else
                    {
                        serviceRet = new ClientResult() { Success = false, Result = null, Message = "无权限调用该方法！" };
                    }
                }
                else
                {
                    serviceRet = new ClientResult() { Success = false, Result = null, Message = "未初始化接口权限验证方法！" };
                }
            }
            catch (ServiceException ex)
            {
                serviceRet = new ClientResult() { Success = false, Message = ex.Message };
                if (ex.InnerException != null)
                {
                    ElincloudServer.Common.Utils.LogUtil.RecordErrorLog(GetExceptionMsg(ex.Message), ex.InnerException);
                }
                else
                {
                    ElincloudServer.Common.Utils.LogUtil.RecordErrorLog(GetExceptionMsg(ex.Message));
                }
            }
            catch (Exception ex)
            {
                serviceRet = new ClientResult() { Success = false, Message = ex.Message };
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog(GetExceptionMsg(ex.Message), ex);
            }

            if (!serviceRet.Success)
            {
                serviceRet.Result = 200;
            }
            return new JsonResult(serviceRet);
        }

        /// <summary>
        /// 不需要登录验证的接口使用此方法
        /// </summary>
        /// <param name="action"></param>
        /// <param name="exceptionMsg"></param>
        /// <returns></returns>
        internal JsonResult Excute(Func<object> action, string exceptionMsg = "")
        {
            try
            {
                var ret = action();
                JsonResult jsonResult = null;
                if (ret is ClientResult)
                {
                    ClientResult cr = ret as ClientResult;
                    if (!cr.Success)
                    {
                        cr.Result = 200;
                    }
                    jsonResult = new JsonResult(cr);
                }
                else
                {
                    jsonResult = new JsonResult(new ClientResult()
                    {
                        Success = true,
                        Result = ret
                    });
                }
                return jsonResult;
            }
            catch (ServiceException ex)
            {
                //serviceRet = new ClientResult() { Success = false, Message = ex.Message };
                if (ex.InnerException != null)
                {
                    ElincloudServer.Common.Utils.LogUtil.RecordErrorLog(GetExceptionMsg(ex.Message), ex.InnerException);
                }
                else
                {
                    ElincloudServer.Common.Utils.LogUtil.RecordErrorLog(GetExceptionMsg(ex.Message));
                }
                return new JsonResult(new ClientResult() { Success = false, Message = ex.Message, Result = 200 });
            }
            catch (Exception ex)
            {
                //serviceRet = new ClientResult() { Success = false, Message = ex.Message };
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog(GetExceptionMsg(ex.Message), ex);
                return new JsonResult(new ClientResult() { Success = false, Message = exceptionMsg != string.Empty ? exceptionMsg : ex.Message, Result = 200 });
            }
        }

        internal JsonResult Excute(Action action, string exceptionMsg = "")
        {
            ClientResult serviceRet = null;
            try
            {
                action();
                JsonResult jsonResult = jsonResult = new JsonResult(new ClientResult()
                {
                    Success = true
                });
                return jsonResult;
            }
            catch (ServiceException ex)
            {
                //serviceRet = new ClientResult() { Success = false, Message = ex.Message };
                serviceRet = new ClientResult() { Success = false, Message = ex.Message, Result = 200 };
                if (ex.InnerException != null)
                {
                    ElincloudServer.Common.Utils.LogUtil.RecordErrorLog(GetExceptionMsg(ex.Message), ex.InnerException);
                }
                else
                {
                    ElincloudServer.Common.Utils.LogUtil.RecordErrorLog(GetExceptionMsg(ex.Message));
                }
            }
            catch (Exception ex)
            {
                serviceRet = new ClientResult() { Success = false, Message = ex.Message, Result = 200 };
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog(GetExceptionMsg(ex.Message), ex);
            }
            return new JsonResult(serviceRet);
        }


        #endregion

        private string GetExceptionMsg(string key)
        {
            if (!string.IsNullOrEmpty(key))
            {
                if (Common.ApplicationEx.GetDicErrors().ContainsKey(key))
                {
                    return Common.ApplicationEx.GetDicErrors()[key];
                }
                else
                {
                    return key;
                }
            }
            else
            {
                return key;
            }
        }
    }
}