
namespace ElincloudServer.Common
{
    /// <summary>
    /// 实用工具类，用于执行带有错误处理和日志记录的操作和函数。
    /// </summary>
    public class ExcuteUtil
    {
        /// <summary>
        /// 执行函数，若发生异常则返回默认值。
        /// </summary>
        /// <typeparam name="TResult">函数返回结果的类型。</typeparam>
        /// <param name="action">要执行的函数。</param>
        /// <param name="exMsg">异常消息。</param>
        /// <returns>函数执行结果或默认值。</return>s
        public static TResult ExcuteOrDefault<TResult>(Func<TResult> action, string exMsg = "")
        {
            try
            {
                return action();
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog(exMsg, ex);
                return default(TResult);
            }
        }

    }
}
