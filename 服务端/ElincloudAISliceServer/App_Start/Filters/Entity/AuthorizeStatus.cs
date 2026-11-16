using System.ComponentModel;

namespace ElincloudServer
{
    public enum AuthorizeStatus
    {
        [Description("E00000452")]
        Success,
        [Description("E00000453")]
        UnLogin,
        [Description("E00000454")]
        TokenInvalid,
        [Description("E00000455")]
        KeyInvalid,
        [Description("E00000456")]
        LoginOut,
        [Description("E00000457")]
        TimeOutOfSync,
        [Description("E00000458")]
        RequestTooBusy
    }
}
