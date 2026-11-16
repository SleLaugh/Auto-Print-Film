
namespace ElincloudServer.Common
{
    public class ThreadUtil
    {
        public void StartThread(Action action)
        {
            if (action != null)
            {
                Thread thread = new Thread(new ThreadStart(action));
                thread.Start();
            }
        }
        private readonly object ThreadLocker = new object();

        public void RunTask(Action action, Action<Task> callBack = null)
        {
            lock (ThreadLocker)
            {
                Task task = Task.Run(action);
                if (callBack != null)
                {
                    callBack(task);
                }
                //这里的sleep为防止外部代码在一个for循环将资源全部占用的问题
                Thread.Sleep(1);
            }

        }
    }
}
