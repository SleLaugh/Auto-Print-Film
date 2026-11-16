
namespace ElincloudServer.Common
{
    public class GZipFileInfo
    {
        public int Index = 0;
        public string RelativePath = null;
        public DateTime ModifiedDate;
        public long Length = 0;
        public bool ParseFileInfo(string fileInfo)
        {
            bool success = false;
            try
            {
                if (!string.IsNullOrEmpty(fileInfo))
                {
                    // get the file information
                    string[] info = fileInfo.Split(',');
                    if (info != null && info.Length == 4)
                    {
                        this.Index = Convert.ToInt32(info[0]);
                        this.RelativePath = info[1].Replace("/", Environment.OSVersion.Platform == PlatformID.Unix ? "/" : "\\");
                        this.ModifiedDate = Convert.ToDateTime(info[2]);
                        this.Length = long.Parse(info[3]);
                        success = true;
                    }
                }
            }
            catch
            {
                success = false;
            }
            return success;
        }
    }
}
