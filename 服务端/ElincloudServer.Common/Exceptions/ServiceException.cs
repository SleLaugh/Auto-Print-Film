
namespace ElincloudServer.Common
{
    public class ServiceException : Exception
    {
        public Exception InnerException { get; set; }
        public ServiceException() : base("服务器调用异常!") { }
        public ServiceException(string msg) : base(msg) { }
        public ServiceException(string msg, Exception innerException) : base(msg) { InnerException = innerException; }

    }
}
