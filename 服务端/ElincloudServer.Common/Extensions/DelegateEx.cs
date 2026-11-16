
namespace ElincloudServer.Common
{
    public static class DelegateEx
    {
        public delegate void SimpleAction();
        public delegate void ParameterAction(object obj);
        public delegate object ParameterReturnAction(object obj);

    }
}
    