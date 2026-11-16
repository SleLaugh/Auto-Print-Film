
using ElincloudServer.Entity;

namespace ElincloudServer.Common
{
    public interface IOSS
    {
        bool GetFileFromOss(string serviceRelaFileName, string localFolder, bool isDecrypt,bool isDecompress = true);
        long PutFolderToOss(string localFolder, string serviceRelaFileName, bool isEncrypt);
        long PutFileToOss(FileInfo localFileInfo, string serviceRelaFileName, bool isEncrypt = true);
        bool DeleteFileFromOss(string serviceRelaFileName);
        List<OSSFileInfo> ListFilesFromOss(string prefix);
        string UploadFileAndGetFileURL(string serviceRelaFileName, string localFileFullName, int Days = 7);
    }
}
