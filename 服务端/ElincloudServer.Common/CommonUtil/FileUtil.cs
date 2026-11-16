using System.Diagnostics;
using Dicom.Imaging;
using Mono.Unix;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace ElincloudServer.Common
{
    public class FileUtil
    {
        #region 连接共享共享文件夹
        public void ConnectShareFolder(string userName, string password, string path)
        {
            if (ElincloudServer.Common.Utils.ConfigUtil.IsWindow)
            {
                if (!string.IsNullOrEmpty(userName) && !string.IsNullOrEmpty(password))
                {
                    string command = $"use \"{path}\" /user:{userName} \"{password}\"";
                    runCmd("net", command);
                }
            }
        }
        private void runCmd(string exe, string args)
        {
            var psi = new ProcessStartInfo
            {
                FileName = exe,
                Arguments = args,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };

            using var proc = Process.Start(psi);
            proc.WaitForExit();

            if (proc.ExitCode != 0)
            {
                string error = proc.StandardError.ReadToEnd();
                Utils.LogUtil.RecordErrorLog("连接共享文件夹失败！" + error);
            }
        }
        #endregion

        public void CreateDirectory(string path)
        {
            if (!Directory.Exists(path))
            {
                Directory.CreateDirectory(path);
                if (Environment.OSVersion.Platform == PlatformID.Unix)
                {
                    SetDirectoryPermissions(path);
                }
            }
        }

        private void SetDirectoryPermissions(string path)
        {
            try
            {
                var dirInfo = new UnixDirectoryInfo(path);
                dirInfo.FileAccessPermissions =
                    FileAccessPermissions.UserReadWriteExecute |
                    FileAccessPermissions.GroupWrite |
                    FileAccessPermissions.OtherReadWriteExecute;

                string parentPath = Directory.GetParent(path)?.FullName ?? string.Empty;
                if (!string.IsNullOrWhiteSpace(parentPath))
                {
                    SetDirectoryPermissions(parentPath);
                }
            }
            catch (Exception ex)
            {
                Utils.LogUtil.RecordErrorLog("设置文件夹权限时异常！", ex);
            }
        }

        public void WriteToText(string Content, string FilePath, string FileName)
        {
            if (!Directory.Exists(FilePath))
            {
                Directory.CreateDirectory(FilePath);//创建新路径
            }

            string Path = Utils.StringUtil.Combine(FilePath, FileName);
            if (!File.Exists(Path))//提出数据存放文件txt   
            {
                FileStream fs;
                fs = File.Create(Path);
                fs.Close();
            }

            using (StreamWriter sw = File.CreateText(Path))
            {
                sw.WriteLine(Content);
                sw.Close();
            }
        }

        public System.Text.Encoding GetFileEncodeType(string filename)
        {
            System.IO.FileStream fs = new System.IO.FileStream(filename, System.IO.FileMode.Open, System.IO.FileAccess.Read);
            System.IO.BinaryReader br = new System.IO.BinaryReader(fs); Byte[] buffer = br.ReadBytes(2);
            if (buffer[0] >= 0xEF)
            {
                if (buffer[0] == 0xEF && buffer[1] == 0xBB)
                {
                    return System.Text.Encoding.UTF8;
                }
                else if (buffer[0] == 0xFE && buffer[1] == 0xFF)
                {
                    return System.Text.Encoding.BigEndianUnicode;
                }
                else if (buffer[0] == 0xFF && buffer[1] == 0xFE)
                {
                    return System.Text.Encoding.Unicode;
                }
                else
                {
                    return System.Text.Encoding.Default;
                }
            }
            else
            {
                return System.Text.Encoding.Default;
            }
        }

        /// <summary>
        /// 删除目录下的文件
        /// </summary>
        /// <param name="folderName"></param>
        public bool DeleteFolder(string folderName, int day = -1)
        {
            try
            {
                //判断文件夹是否还存在
                if (Directory.Exists(folderName))
                {
                    //去除文件的只读属性
                    System.IO.File.SetAttributes(folderName, System.IO.FileAttributes.Normal);
                    foreach (string f in Directory.GetFileSystemEntries(folderName))
                    {
                        if (File.Exists(f))
                        {
                            FileInfo file = new FileInfo(f);
                            if (day >= 0)
                            {
                                if (file.CreationTime.Date <= DateTime.Now.AddDays(-day).Date)
                                {
                                    //如果有子文件删除文件
                                    File.Delete(f);
                                }
                            }
                            else
                            {
                                File.Delete(f);
                            }

                        }
                        else
                        {
                            //循环递归删除子文件夹 
                            DeleteFolder(f, day);
                        }
                    }
                    if (Directory.GetFileSystemEntries(folderName).Length == 0)
                    {
                        Directory.Delete(folderName);
                    }
                }
                return true;
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("删除文件夹异常！" + folderName, ex);
                return false;
            }
        }

        #region 移除只读属性并删除文件

        public bool DeleteFile(string fileName)
        {
            FileInfo fileInfo = new FileInfo(fileName);
            return DeleteFile(fileInfo);
        }

        public bool DeleteFile(FileInfo file)
        {
            try
            {
                if (System.IO.File.Exists(file.FullName))
                {
                    file.Attributes &= ~FileAttributes.ReadOnly; // 移除只读属性
                    System.IO.File.Delete(file.FullName);
                    return true;
                }
                else
                {
                    return false;
                }
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("删除文件异常！", ex);
                return false;
            }
        }

        #endregion

        public string GetTMPFile(string localFileFullName)
        {
            string tmpFileName = string.Empty;
            try
            {
                //Linux 系统环境变量里没有  TMP
                //string tmpPath = System.Environment.GetEnvironmentVariable("TMP");
                string tmpPath = Path.GetTempPath();
                tmpPath = ElincloudServer.Common.Utils.StringUtil.Combine(tmpPath, ElincloudServer.Common.ApplicationEx.TMPFolder, Guid.NewGuid().ToString());
                System.IO.Directory.CreateDirectory(tmpPath);
                tmpFileName = ElincloudServer.Common.Utils.StringUtil.Combine(tmpPath, Utils.StringUtil.GetFileName(localFileFullName));
                System.IO.File.Copy(localFileFullName, tmpFileName);
                return tmpFileName;
            }
            catch (Exception ex)
            {
                string exMessage = "复制文件[" + localFileFullName + "]到临时目录异常！";
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog(exMessage, ex);
                //throw new Exception(exMessage, ex);
            }
            return tmpFileName;
        }

        public void ClearTMPFiles()
        {
            try
            {
                //Linux 系统环境变量里没有  TMP
                //string tmpPath = System.Environment.GetEnvironmentVariable("TMP") ;
                string tmpPath = Path.GetTempPath();
                tmpPath = ElincloudServer.Common.Utils.StringUtil.Combine(tmpPath, ApplicationEx.TMPFolder);
                if (System.IO.Directory.Exists(tmpPath))
                {
                    System.IO.Directory.Delete(tmpPath, true);
                }
            }
            catch (Exception ex)
            {
                string exMessage = "清空临时目录异常！";
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog(exMessage, ex);
            }
        }

        public bool MoveFile(string sourceFileName, string destFileName)
        {
            try
            {
                if (!System.IO.File.Exists(sourceFileName))
                {
                    throw new Exception("没有找到要移动的文件[" + sourceFileName + "]!");
                }
                if (System.IO.File.Exists(destFileName))
                {
                    System.IO.File.Delete(destFileName);
                }
                System.IO.File.Move(sourceFileName, destFileName);
                return true;
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("移动文件异常！", ex);
                return false;
            }

        }

        public bool CopyDirectory(string SourcePath, string DestinationPath, bool overwriteexisting = true)
        {
            bool ret = false;
            try
            {
                if (Directory.Exists(SourcePath))
                {
                    if (!Directory.Exists(DestinationPath))
                    {
                        Directory.CreateDirectory(DestinationPath);
                    }

                    // 复制文件
                    foreach (string fls in Directory.GetFiles(SourcePath))
                    {
                        FileInfo flinfo = new FileInfo(fls);
                        flinfo.CopyTo(Utils.StringUtil.Combine(DestinationPath, flinfo.Name), overwriteexisting);
                    }

                    // 递归复制子目录
                    foreach (string drs in Directory.GetDirectories(SourcePath))
                    {
                        DirectoryInfo drinfo = new DirectoryInfo(drs);
                        if (CopyDirectory(drs, Utils.StringUtil.Combine(DestinationPath, drinfo.Name), overwriteexisting) == false)
                            ret = false;
                    }

                    ret = true;
                }
                else
                {
                    ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("要复制的文件夹[" + SourcePath + "]不存在！");
                }
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("复制文件夹异常！", ex);
                ret = false;
            }
            return ret;
        }


        public string FormatFileSize(long fileSize)
        {
            string fileSizeStr = string.Empty;
            if (fileSize < 0)
            {
                fileSizeStr = "0B";
            }
            else if (fileSize < 1024 * 1024)
            {
                fileSizeStr = string.Format("{0}KB", (Math.Round(fileSize / (1024.0), 2)).ToString("0.00"));
            }
            else
            {
                fileSizeStr = string.Format("{0}MB", (Math.Round(fileSize / (1024 * 1024.0), 2)).ToString("0.00"));
            }
            return fileSizeStr;
        }

        public string ConvertDicomFileToBase64ThumbnailImage(string dicomFileFullName, int width = 100, int height = 100)
        {
            try
            {
                DicomImage di = new DicomImage(dicomFileFullName);
                var a = di.RenderImage().AsBytes();
                Image<Rgba32> image = Image.LoadPixelData<Rgba32>(a, di.Width, di.Height);
                image.Mutate(x => x.Resize(new ResizeOptions()
                {
                    Size = new SixLabors.ImageSharp.Size(width, height),
                    Mode = ResizeMode.Max
                }));
                string base64Str = image.ToBase64String(SixLabors.ImageSharp.Formats.Jpeg.JpegFormat.Instance);
                return base64Str;
            }
            catch (Exception ex)
            {
                return null;
            }
        }
    }
}
