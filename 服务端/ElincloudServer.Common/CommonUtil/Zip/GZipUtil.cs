using System.IO.Compression;
using System.Text;
using ICSharpCode.SharpZipLib.Zip;

namespace ElincloudServer.Common
{
    public class GZipUtil
    {
        /// <summary>
        /// 缓冲区大小（默认300KB）
        /// </summary>
        private static int bufferSize = 1024 * 300;

        /// <summary>
        /// 构造函数
        /// </summary>
        /// <remarks>
        /// 注册编码提供程序，用于解决ZIP中文文件名乱码问题
        /// （适用于 .NET Core / .NET 5+）
        /// </remarks>
        public GZipUtil()
        {
            //这个方法是为了解决文件名中文乱码问题
            Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
        }

        #region 公开方法

        /// <summary>
        /// 压缩文件夹
        /// </summary>
        /// <param name="sourceFolder">源目录路径</param>
        /// <param name="destFolder">目标目录（存放压缩包）</param>
        /// <param name="zipFileName">压缩文件名（如：a.zip）</param>
        /// <param name="isEncrypt">是否使用加密压缩（自定义格式）</param>
        /// <returns>压缩结果</returns>
        /// <remarks>
        /// 会递归获取所有文件并进行压缩
        /// </remarks>
        public GZipResult Compress(string sourceFolder, string destFolder, string zipFileName, bool isEncrypt)
        {
            DirectoryInfo di = new DirectoryInfo(sourceFolder);
            FileInfo[] files = di.GetFiles("*.*", SearchOption.AllDirectories);
            return Compress(files, sourceFolder, destFolder, zipFileName, isEncrypt);
        }

        /// <summary>
        /// 压缩文件集合
        /// </summary>
        /// <param name="files">待压缩文件集合</param>
        /// <param name="sourceFolder">源目录（用于生成相对路径）</param>
        /// <param name="destFolder">目标目录</param>
        /// <param name="zipFileName">压缩文件名</param>
        /// <param name="isEncrypt">是否使用加密压缩（自定义格式）</param>
        /// <returns>压缩结果</returns>
        /// <remarks>
        /// 根据 isEncrypt 选择不同压缩方式：
        /// false → 标准ZIP
        /// true → 自定义GZip加密格式
        /// </remarks>
        public GZipResult Compress(FileInfo[] files, string sourceFolder, string destFolder, string zipFileName, bool isEncrypt)
        {
            GZipResult result = new GZipResult();
            try
            {
                if (files.Length > 0)
                {
                    if (!System.IO.Directory.Exists(destFolder))
                    {
                        Common.Utils.FileUtil.CreateDirectory(destFolder);
                    }
                    string zipFileFullName = ElincloudServer.Common.Utils.StringUtil.Combine(destFolder, zipFileName);
                    if (System.IO.File.Exists(zipFileFullName))
                    {
                        System.IO.File.Delete(zipFileFullName);
                    }
                    if (isEncrypt)
                    {
                        zipEncryptFile(files, sourceFolder, zipFileFullName, result);
                    }
                    else
                    {
                        zipFile(files, sourceFolder, zipFileFullName, result);
                    }
                }
                else
                {
                    result.ErrorMessage = "没有要压缩的文件！";
                    result.Errors = true;
                }
            }
            catch (Exception ex)
            {
                result.ErrorMessage = ex.Message;
                result.Errors = true;
            }
            return result;
        }

        /// <summary>
        /// 解压缩文件
        /// </summary>
        /// <param name="zipFileFullName">压缩文件完整路径</param>
        /// <param name="destFolder">解压目标目录</param>
        /// <param name="isDecrypt">是否为加密压缩文件</param>
        /// <param name="password">解压密码（仅对标准ZIP有效）</param>
        /// <returns>解压结果</returns>
        /// <remarks>
        /// 根据 isDecrypt 选择解压方式：
        /// false → 标准ZIP解压
        /// true → 自定义加密格式解压
        /// </remarks>
        public GZipResult Decompress(string zipFileFullName, string destFolder, bool isDecrypt, string password = "")
        {
            if (isDecrypt)
            {
                return unZipEncryptFile(zipFileFullName, destFolder);
            }
            else
            {
                return unZipFile(zipFileFullName, destFolder, password);
            }
        }

        #endregion

        #region 加密压缩与解压缩

        /// <summary>
        /// 加密压缩文件（自定义GZip格式，非标准ZIP）
        /// </summary>
        /// <param name="files">待压缩文件集合</param>
        /// <param name="sourceFolder">源目录（用于生成相对路径）</param>
        /// <param name="zipFileFullName">输出文件完整路径</param>
        /// <param name="result">压缩结果对象</param>
        /// <remarks>
        /// 通过写入自定义头信息（文件索引、相对路径、时间、大小）实现多文件打包
        /// 注意：生成文件仅能通过本工具解压，不兼容标准ZIP工具
        /// </remarks>
        private void zipEncryptFile(FileInfo[] files, string sourceFolder, string zipFileFullName, GZipResult result)
        {
            //以下流对象关闭的顺序很重要 
            //1.无损压缩（先关闭gzip再关闭fsout可以保证无损压缩）
            //2.有损压缩（先关闭fsout，后面gzip会自动关闭）
            try
            {
                result.ZipFile = zipFileFullName;
                result.FileCount = files.Length;
                // open the temp file for writing
                using (FileStream fsOut = new FileStream(zipFileFullName, FileMode.Create, FileAccess.Write, FileShare.None))
                {
                    using (GZipStream gzip = new GZipStream(fsOut, CompressionMode.Compress))
                    {
                        for (int i = 0; i < files.Length; i++)
                        {
                            FileInfo fileInfo = files[i];
                            string relaFileFullName = fileInfo.FullName.Replace(sourceFolder, string.Empty).Replace("\\", "/");
                            if (relaFileFullName.StartsWith("/"))
                            {
                                relaFileFullName = relaFileFullName.Substring(1, relaFileFullName.Length - 1);
                            }
                            //写入加密头文件
                            string fileHeader = i.ToString() + "," + relaFileFullName + "," + fileInfo.LastWriteTimeUtc.ToString() + "," + fileInfo.Length.ToString() + "\n";
                            byte[] header = Encoding.Default.GetBytes(fileHeader);
                            gzip.Write(header, 0, header.Length);
                            //写入文件内容
                            using (FileStream fsIn = new FileStream(fileInfo.FullName, FileMode.Open, FileAccess.Read, FileShare.Read))
                            {
                                int count = 0;
                                byte[] subBuffer = new byte[bufferSize];
                                while ((count = fsIn.Read(subBuffer, 0, bufferSize)) > 0)
                                {
                                    gzip.Write(subBuffer, 0, count);
                                }
                            }
                            //写入加密文件结尾 以换行符 \n 为文件结尾
                            gzip.WriteByte(10);
                        }
                    }
                }
                result.ZipFileSize = new FileInfo(zipFileFullName).Length;
            }
            catch (Exception ex)
            {
                result.ErrorMessage = ex.Message;
                result.Errors = true;
                //压缩失败时，删除生成的压缩文件
                if (System.IO.File.Exists(zipFileFullName))
                {
                    System.IO.File.Delete(zipFileFullName);
                }
            }
        }

        /// <summary>
        /// 解压加密压缩文件（对应 zipEncryptFile）
        /// </summary>
        /// <param name="zipFileFullName">加密压缩文件路径</param>
        /// <param name="destFolder">解压目标目录</param>
        /// <returns>解压结果（包含文件数量、错误信息等）</returns>
        /// <remarks>
        /// 按自定义头逐个解析文件并写入本地
        /// </remarks>
        private GZipResult unZipEncryptFile(string zipFileFullName, string destFolder)
        {
            GZipResult result = new GZipResult();
            try
            {
                result.ZipFile = zipFileFullName;
                if (System.IO.File.Exists(zipFileFullName))
                {
                    using (FileStream fsIn = new FileStream(zipFileFullName, FileMode.Open, FileAccess.Read, FileShare.Read))
                    {
                        result.ZipFileSize = fsIn.Length;
                        using (GZipStream gzip = new GZipStream(fsIn, CompressionMode.Decompress, true))
                        {
                            while (gzip.BaseStream.Position != gzip.BaseStream.Length)
                            {
                                string line = string.Empty;
                                while (string.IsNullOrEmpty(line) && gzip.BaseStream.Position != gzip.BaseStream.Length)
                                {
                                    line = readHeader(gzip);
                                }
                                if (!string.IsNullOrEmpty(line))
                                {
                                    GZipFileInfo gzf = new GZipFileInfo();
                                    if (gzf.ParseFileInfo(line) && gzf.Length > 0)
                                    {
                                        string unZipSingleFileFullName = ElincloudServer.Common.Utils.StringUtil.Combine(destFolder, gzf.RelativePath);
                                        string unZipSingleFileFolder = Utils.StringUtil.GetDirectoryName(unZipSingleFileFullName);
                                        // make sure the folder exists
                                        if (!Directory.Exists(unZipSingleFileFolder))
                                        {
                                            Common.Utils.FileUtil.CreateDirectory(unZipSingleFileFolder);
                                        }
                                        // read from fsTemp and write out the file
                                        writeFile(gzip, gzf.Length, unZipSingleFileFullName);
                                        result.FileCount += 1;
                                    }
                                }
                            }
                        }
                    }
                }
                else
                {
                    result.ErrorMessage = "找不到要解压的文件[" + zipFileFullName + "]";
                    result.Errors = true;
                }
            }
            catch (Exception ex)
            {
                result.ErrorMessage = ex.Message;
                result.Errors = true;
            }
            return result;
        }

        /// <summary>
        /// 读取加密文件头信息
        /// </summary>
        /// <param name="stream">GZip解压流</param>
        /// <returns>解析出的头字符串（包含文件信息）</returns>
        /// <remarks>
        /// 头信息以换行符(\n)结束，格式：
        /// index,relativePath,lastWriteTime,length
        /// </remarks>
        private static string readHeader(GZipStream stream)
        {
            byte[] buffer = new byte[bufferSize];
            byte b = 0;
            byte lf = 10;
            int i = 0;
            while (b != lf)
            {
                b = (byte)stream.ReadByte();
                buffer[i] = b;
                i++;
                //读取头文件的时候可能有以下情况（头文件内容长度不可能把整个buffer占满）：
                //1.有时候stream.ReadByte()实际上已经读完但stream.BaseStream.Position并没有推到结尾
                //2.有时候stream.ReadByte()没读完但stream.BaseStream.Position已经推到结尾
                //所以用下面的这个判断来确保当buffer已经读满的情况下就不再读了直接返回
                if (i >= bufferSize)
                {
                    break;
                }
            }
            //return System.Text.Encoding.Default.GetString(buffer, 0, i - 1);

            string retStr = System.Text.Encoding.Default.GetString(buffer, 0, i - 1);
            //移除首尾255空格
            byte[] bytes = { 255 };
            char[] chars = System.Text.Encoding.Default.GetChars(bytes, 0, bytes.Length);
            retStr = retStr.Trim(chars);
            return retStr;
        }

        /// <summary>
        /// 从压缩流中读取指定长度数据并写入文件
        /// </summary>
        /// <param name="fs">输入流（解压后的GZip流）</param>
        /// <param name="fileLength">当前文件长度</param>
        /// <param name="fileFullName">输出文件完整路径</param>
        /// <returns>是否写入成功</returns>
        /// <remarks>
        /// 按指定长度读取，避免读串数据（多个文件拼接在同一流中）
        /// </remarks>
        private static bool writeFile(Stream fs, long fileLength, string fileFullName)
        {
            bool success = false;
            try
            {
                using (FileStream fsFile = new FileStream(fileFullName, FileMode.Create, FileAccess.Write, FileShare.None))
                {
                    byte[] buffer = new byte[bufferSize];
                    long remainSize = fileLength;
                    int readSize = 0;
                    int count = 0;
                    while (remainSize > 0)
                    {
                        if (remainSize > bufferSize)
                        {
                            readSize = bufferSize;
                        }
                        else
                        {
                            readSize = (int)remainSize;
                        }
                        count = fs.Read(buffer, 0, readSize);
                        remainSize -= count;
                        if (count > 0)
                        {
                            fsFile.Write(buffer, 0, count);
                            //fsFile.Flush();
                        }
                        else
                        {
                            break;
                        }
                    }
                    fsFile.Flush();
                    success = true;
                }
            }
            catch //(Exception ex2)
            {
                success = false;
            }
            return success;
        }

        #endregion

        #region 常规压缩与解压缩（zip）

        /// <summary>
        /// 普通ZIP压缩（无进度回调）
        /// </summary>
        /// <param name="files">待压缩文件集合</param>
        /// <param name="sourceFolder">源目录（用于生成相对路径）</param>
        /// <param name="zipFileFullName">压缩包完整路径（如：d:\a.zip）</param>
        /// <param name="result">压缩结果对象</param>
        private void zipFile(FileInfo[] files, string sourceFolder, string zipFileFullName, GZipResult result)
        {
            try
            {
                result.FileCount = files.Length;
                result.ZipFile = zipFileFullName;
                using (ZipOutputStream stream = new ZipOutputStream(File.Create(zipFileFullName)))
                {
                    stream.SetLevel(1); // 压缩级别 0-9 
                    //stream.Password = password; //Zip压缩文件密码
                    foreach (FileInfo file in files)
                    {
                        string relaFileFullName = file.FullName.Replace(sourceFolder, string.Empty).Replace("\\", "/");
                        if (relaFileFullName.StartsWith("/"))
                        {
                            relaFileFullName = relaFileFullName.Substring(1, relaFileFullName.Length - 1);
                        }
                        ZipEntry entry = new ZipEntry(relaFileFullName);
                        entry.DateTime = DateTime.Now;
                        stream.PutNextEntry(entry);
                        using (FileStream fs = File.OpenRead(file.FullName))
                        {
                            int count = 0;
                            byte[] subBuffer = new byte[bufferSize];
                            while ((count = fs.Read(subBuffer, 0, bufferSize)) > 0)
                            {
                                stream.Write(subBuffer, 0, count);
                            }
                        }
                    }
                    stream.Finish();
                }
                result.ZipFileSize = new FileInfo(zipFileFullName).Length;
            }
            catch (Exception ex)
            {
                result.ErrorMessage = ex.Message;
                result.Errors = true;
                //压缩失败时，删除生成的压缩文件
                if (System.IO.File.Exists(zipFileFullName))
                {
                    System.IO.File.Delete(zipFileFullName);
                }
            }
        }

        /// <summary>
        /// 压缩进度回调委托
        /// </summary>
        /// <param name="totalBytes">总字节数</param>
        /// <param name="processedBytes">已处理字节数</param>
        public delegate void CompressionProgressCallback(long totalBytes, long processedBytes);

        /// <summary>
        /// ZIP压缩（带进度回调）
        /// </summary>
        /// <param name="sourceFolder">源目录（如：d:\test）</param>
        /// <param name="zipFileFullName">压缩包完整路径（如：d:\a.zip）</param>
        /// <param name="progressCallback">进度回调（返回总字节数与当前已处理字节数）</param>
        /// <returns>压缩结果（包含文件数、大小、错误信息等）</returns>
        public GZipResult zipFileWithProgress(string sourceFolder, string zipFileFullName, CompressionProgressCallback progressCallback)
        {
            GZipResult result = new GZipResult();

            try
            {
                if (!Directory.Exists(sourceFolder))
                {
                    result.Errors = true;
                    result.ErrorMessage = "源目录不存在";
                    return result;
                }

                var files = Directory.GetFiles(sourceFolder, "*.*", SearchOption.AllDirectories);

                long totalBytes = 0;
                foreach (var f in files)
                {
                    totalBytes += new FileInfo(f).Length;
                }

                long processedBytes = 0;
                long lastReport = 0;

                result.FileCount = files.Length;
                result.ZipFile = zipFileFullName;

                using (var zipStream = new ZipOutputStream(File.Create(zipFileFullName)))
                {
                    zipStream.SetLevel(1);

                    foreach (var file in files)
                    {
                        FileInfo fileInfo = new FileInfo(file);

                        string entryName = Path.GetRelativePath(sourceFolder, file)
                                               .Replace("\\", "/");

                        ZipEntry entry = new ZipEntry(entryName)
                        {
                            DateTime = fileInfo.LastWriteTime,
                            Size = fileInfo.Length
                        };

                        zipStream.PutNextEntry(entry);

                        using (FileStream fs = File.OpenRead(file))
                        {
                            byte[] buffer = new byte[4096];
                            int bytesRead;

                            while ((bytesRead = fs.Read(buffer, 0, buffer.Length)) > 0)
                            {
                                zipStream.Write(buffer, 0, bytesRead);

                                processedBytes += bytesRead;

                                // 节流（200KB触发一次）
                                if (progressCallback != null && processedBytes - lastReport > 1024 * 200)
                                {
                                    lastReport = processedBytes;
                                    progressCallback(totalBytes, processedBytes);
                                }
                            }
                        }

                        zipStream.CloseEntry();
                    }

                    zipStream.Finish();
                }

                // 最后补一次100%
                if (progressCallback != null)
                {
                    progressCallback(totalBytes, totalBytes);
                }

                result.ZipFileSize = new FileInfo(zipFileFullName).Length;
            }
            catch (Exception ex)
            {
                result.Errors = true;
                result.ErrorMessage = ex.Message;

                if (File.Exists(zipFileFullName))
                {
                    File.Delete(zipFileFullName);
                }
            }

            return result;
        }

        /// <summary>
        /// 解压ZIP文件
        /// </summary>
        /// <param name="zipFileFullName">压缩包完整路径（如：d:\a.zip）</param>
        /// <param name="destFolder">解压目标目录</param>
        /// <param name="password">解压密码（无密码可传空）</param>
        /// <returns>解压结果（包含文件数、大小、错误信息等）</returns>
        private GZipResult unZipFile(string zipFileFullName, string destFolder, string password = "")
        {
            GZipResult result = new GZipResult();
            try
            {
                result.ZipFile = zipFileFullName;
                if (System.IO.File.Exists(zipFileFullName))
                {
                    using (FileStream fsIn = File.OpenRead(zipFileFullName))
                    {
                        result.ZipFileSize = fsIn.Length;
                        using (ZipInputStream s = new ZipInputStream(fsIn))
                        {
                            s.Password = password;
                            ZipEntry theEntry;
                            while ((theEntry = s.GetNextEntry()) != null)
                            {
                                //下面的path可能是文件夹，也可能是FileFullName
                                string path = ElincloudServer.Common.Utils.StringUtil.Combine(destFolder, theEntry.Name);
                                // make sure the folder exists
                                if (theEntry.IsDirectory)
                                {
                                    Common.Utils.FileUtil.CreateDirectory(path);
                                }
                                else if (theEntry.IsFile)
                                {
                                    using (FileStream fsOut = File.Create(path))
                                    {
                                        int count = 0;
                                        byte[] subBuffer = new byte[bufferSize];
                                        while ((count = s.Read(subBuffer, 0, bufferSize)) > 0)
                                        {
                                            fsOut.Write(subBuffer, 0, count);
                                        }
                                    }
                                    result.FileCount += 1;
                                }
                            }
                        }
                    }
                }
                else
                {
                    result.ErrorMessage = "找不到要解压的文件[" + zipFileFullName + "]";
                    result.Errors = true;
                }
            }
            catch (Exception ex)
            {
                result.ErrorMessage = ex.Message;
                result.Errors = true;
            }
            return result;
        }

        #endregion

    }
}

