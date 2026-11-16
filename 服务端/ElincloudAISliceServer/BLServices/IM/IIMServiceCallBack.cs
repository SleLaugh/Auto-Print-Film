using ElincloudServer.Entity;

namespace ElincloudServer.BLServices
{
    /// <summary>
    /// 消息类接口集合
    /// </summary>
    public interface IIMServiceCallBack
    {
        /// <summary>
        /// 消息发送
        /// </summary>
        /// <param name="message">消息内容</param>
        void SendMessage(SysMessageInfo message);

        /// <summary>
        /// 当教学教室的在线用户列表发生变化的时候
        /// </summary>
        //void OnCourseRoomUserListChanged(CourseInfo course, int count);

        /// <summary>
        /// 当会诊室的在线用户列表发生变化的时候
        /// </summary>
        //void OnConsultationRoomUserListChanged(ConsultationInfo consultation, int count);
    }
}
