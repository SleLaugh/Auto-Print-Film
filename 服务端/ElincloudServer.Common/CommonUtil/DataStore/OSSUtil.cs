using Aliyun.OSS;
using Aliyun.OSS.Common;
using Aliyun.OSS.Model;
using Amazon.S3;
using ElincloudServer.Entity;
using MySqlX.XDevAPI;

namespace ElincloudServer.Common
{
    public class OSSUtil : IOSS
    {
        private string _bucketName = string.Empty;
        public string BucketName
        {
            get { return _bucketName; }
        }

        OssClient client = null;

        /// <summary>
        /// 外网链接对象
        /// </summary>
        OssClient externalClient = null;

        //public OSSUtil()
        //{
        //    string accessKeyID = Common.App.RegUtil.GetOSSAccessKeyID();
        //    string accessKeySecret = Common.App.RegUtil.GetOSSAccessKeySecret();
        //    string endPoint = Common.App.RegUtil.GetOSSEndPoint();
        //    _bucketName = Common.App.RegUtil.GetOSSBucketName();
        //    client = new OssClient(endPoint, accessKeyID, accessKeySecret);
        //}

        public OSSUtil(string keyID, string keySecret, string endpoint, string bucketName)
        {
            _bucketName = bucketName;
            client = new OssClient(endpoint, keyID, keySecret);
        }

        public OSSUtil(string keyID, string keySecret, string endpoint, string externalEndPoint, string bucketName)
        {
            _bucketName = bucketName;
            client = new OssClient(endpoint, keyID, keySecret);

            if (!string.IsNullOrEmpty(externalEndPoint))
            {
                client = new OssClient(externalEndPoint, keyID, keySecret);
            }
        }


        /// <summary>
        /// 从对像存储下载文件
        /// </summary>
        /// <param name="serviceRelaFileName">如:@"RISImages\DX\20181115\2221659\000100000010\1.2.840.113619.2.203.4.2147483647.1542300170.371687"</param>
        /// <param name="localFolder">如:@"E:\FTPFiles\RISImages\CT\20181016\00000100000006"</param>
        public bool GetFileFromOss(string serviceRelaFileName, string localFolder, bool isDecrypt, bool isDecompress = true)
        {
            try
            {
                serviceRelaFileName = serviceRelaFileName.Replace('\\', '/');
                while (serviceRelaFileName.Contains("//"))
                {
                    serviceRelaFileName = serviceRelaFileName.Replace("//", "/");
                }
                if (isDecrypt)
                {
                    serviceRelaFileName = serviceRelaFileName + ".oss";
                }
                ObjectMetadata metadata = client.GetObjectMetadata(BucketName, serviceRelaFileName);
                string state = metadata.HttpMetadata["x-oss-storage-class"] as string;
                if (state == "Archive" || state == "ColdArchive")//state=="IA" 低频 
                {
                    try
                    {
                        RestoreObjectResult result = client.RestoreObject(BucketName, serviceRelaFileName);
                        if (result.HttpStatusCode == System.Net.HttpStatusCode.OK)
                        {
                            download(serviceRelaFileName, localFolder, isDecrypt, isDecompress);
                        }
                        else
                        {
                            ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("要下载的文件[" + serviceRelaFileName + "]正在解冻中...");
                            return false;
                        }
                    }
                    catch (Exception ex)
                    {
                        ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("要下载的文件[" + serviceRelaFileName + "]正在解冻中...");
                        return false;
                    }
                }
                else
                {
                    download(serviceRelaFileName, localFolder, isDecrypt, isDecompress);
                }
                return true;
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("从OSS上下载文件[" + serviceRelaFileName + "]异常!", ex);
                return false;
            }

        }

        private void download(string serviceRelaFileName, string localFolder, bool isDecrypt, bool isDecompress = true)
        {
            var result = client.GetObject(BucketName, serviceRelaFileName);
            string localFileName = Utils.StringUtil.GetFileName(serviceRelaFileName);
            string tempFolder = isDecompress ? ElincloudServer.Common.Utils.StringUtil.Combine(localFolder, "temp") : localFolder;//不解压缩时直接到目标目录
            string localFileFullName = ElincloudServer.Common.Utils.StringUtil.Combine(tempFolder, localFileName);
            using (var requestStream = result.Content)
            {
                if (!System.IO.Directory.Exists(tempFolder))
                {
                    Common.Utils.FileUtil.CreateDirectory(tempFolder);
                }
                using (var fs = File.Open(localFileFullName, FileMode.OpenOrCreate))
                {
                    int length = 4 * 1024;
                    var buf = new byte[length];
                    do
                    {
                        length = requestStream.Read(buf, 0, length);
                        fs.Write(buf, 0, length);
                    } while (length != 0);
                }
            }
            if (isDecompress)
            {
                var ret = ElincloudServer.Common.Utils.GZipUtil.Decompress(localFileFullName, localFolder, isDecrypt);
                if (ret.Errors)
                {
                    //ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("解压文件失败!");
                    throw new Exception("解压文件失败!");
                }
                else
                {
                    System.IO.Directory.Delete(tempFolder, true);
                }
            }
        }

        /// <summary>
        /// 加密压缩文件夹并上传
        /// </summary>
        /// <param name="serviceRelaFileName">如:@"RISImages\DX\20181115\2221659\000100000010\1.2.840.113619.2.203.4.2147483647.1542300170.371687.oss"</param>
        /// <param name="localFolder">如:@"E:\FTPFiles\RISImages\CT\20181016\00000100000006"</param>
        public long PutFolderToOss(string localFolder, string serviceRelaFileName, bool isEncrypt)
        {
            try
            {
                string fileName = Utils.StringUtil.GetFileName(serviceRelaFileName);
                //压缩之前先检测下压缩后是否存在同名文件，存在则删除
                string ossFileFullName = ElincloudServer.Common.Utils.StringUtil.Combine(localFolder, fileName);
                if (System.IO.File.Exists(ossFileFullName))
                {
                    System.IO.File.Delete(ossFileFullName);
                }
                Common.GZipResult ret = ElincloudServer.Common.Utils.GZipUtil.Compress(localFolder, localFolder, fileName, isEncrypt);
                if (!ret.Errors)
                {
                    bool uploadSuccess = partUpload(serviceRelaFileName, ossFileFullName);
                    if (uploadSuccess)
                    {
                        System.IO.File.Delete(ret.ZipFile);
                        return ret.ZipFileSize;
                    }
                    else
                    {
                        return -1;
                    }
                }
                else
                {
                    ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("压缩文件夹失败," + ret.ErrorMessage + "![" + localFolder + "]");
                    return -1;
                }

            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("上传文件到对像存储异常!", ex);
                return -1;
            }
        }

        /// <summary>
        /// 单文件上传
        /// </summary>
        /// <param name="localFileInfo"></param>
        /// <param name="serviceRelaFileName"></param>
        /// <param name="isEncrypt">是否加密（采用加密的同时会对文件进行压缩）</param>
        /// <returns></returns>
        public long PutFileToOss(FileInfo localFileInfo, string serviceRelaFileName, bool isEncrypt = true)
        {
            try
            {
                if (localFileInfo.Exists)
                {
                    string fileName = Utils.StringUtil.GetFileName(serviceRelaFileName);
                    string localFolder = Utils.StringUtil.GetDirectoryName(localFileInfo.FullName);
                    if (isEncrypt)
                    {
                        string ossFileFullName = ElincloudServer.Common.Utils.StringUtil.Combine(localFolder, fileName);
                        if (System.IO.File.Exists(ossFileFullName))
                        {
                            System.IO.File.Delete(ossFileFullName);
                        }
                        FileInfo[] files = { localFileInfo };
                        Common.GZipResult ret = ElincloudServer.Common.Utils.GZipUtil.Compress(files, localFolder, localFolder, fileName, isEncrypt);
                        if (!ret.Errors)
                        {
                            bool uploadSuccess = partUpload(serviceRelaFileName, ossFileFullName);
                            if (uploadSuccess)
                            {
                                System.IO.File.Delete(ret.ZipFile);
                                return ret.ZipFileSize;
                            }
                            else
                            {
                                return -1;
                            }
                        }
                        else
                        {
                            ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("压缩文件失败," + ret.ErrorMessage + "![" + localFileInfo.FullName + "]");
                            return -1;
                        }
                    }
                    else
                    {
                        bool uploadSuccess = partUpload(serviceRelaFileName, localFileInfo.FullName);
                        if (uploadSuccess)
                        {
                            return localFileInfo.Length;
                        }
                        else
                        {
                            return -1;
                        }
                    }
                }
                else
                {
                    ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("要上传的文件[" + localFileInfo.FullName + "]不存在！");
                    return -1;
                }
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("上传文件到OSS异常!", ex);
                return -1;
            }
        }

        private bool partUpload(string serviceRelaFileName, string ossFileFullName)
        {
            serviceRelaFileName = serviceRelaFileName.Replace('\\', '/');
            string uploadId = "";
            #region part upload

            // 1. 初始化分段上传任务
            try
            {
                // 定义上传文件的名字和所属存储空间。在InitiateMultipartUploadRequest中，可以设置ObjectMeta，但不必指定其中的ContentLength。
                var request = new InitiateMultipartUploadRequest(BucketName, serviceRelaFileName);
                var result = client.InitiateMultipartUpload(request);
                uploadId = result.UploadId;
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("初始化OSS分片上传存储空间失败!", ex);
                return false;
            }

            // 2. 上传段

            // 开始分片上传。partETags是保存partETag的列表，OSS收到用户提交的分片列表后，会逐一验证每个分片数据的有效性。 当所有的数据分片通过验证后，OSS会将这些分片组合成一个完整的文件。
            var partETags = new List<PartETag>();
            try
            {
                string filePath = ossFileFullName;//上传的本地文件路径，需要指定到具体的文件名
                long contentLength = new FileInfo(filePath).Length;
                // 计算分片总数。
                //var partSize = 5 * 1024 * 1024;
                long partSize = 5242880;//5M
                var fileSize = contentLength;
                var partCount = contentLength / partSize;
                if (fileSize % partSize != 0)
                {
                    partCount++;
                }
                using (var fs = File.Open(ossFileFullName, FileMode.Open))
                {
                    for (var i = 0; i < partCount; i++)
                    {
                        var skipBytes = (long)partSize * i;
                        // 定位到本次上传起始位置。
                        fs.Seek(skipBytes, 0);
                        // 计算本次上传的片大小，最后一片为剩余的数据大小。
                        var size = (partSize < fileSize - skipBytes) ? partSize : (fileSize - skipBytes);
                        var request = new UploadPartRequest(BucketName, serviceRelaFileName, uploadId)
                        {
                            InputStream = fs,
                            PartSize = size,
                            PartNumber = i + 1
                        };
                        // 调用UploadPart接口执行上传功能，返回结果中包含了这个数据片的ETag值。
                        var result = client.UploadPart(request);
                        partETags.Add(result.PartETag);
                        //Common.App.LogUtil.RecordRunLog(string.Format("finish {0}/{1}", partETags.Count, partCount));
                    }
                    //Common.App.LogUtil.RecordRunLog("Put multi part upload succeeded");
                }
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("分片上传异常!", ex);
                //取消分段上传任务
                try
                {
                    AbortMultipartUploadRequest request = new AbortMultipartUploadRequest(BucketName, serviceRelaFileName, uploadId);
                    client.AbortMultipartUpload(request);
                }
                catch (Exception e)
                {
                    ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("取消分段上传任务异常!", e);
                }
                return false;
            }
            // 完成分片上传。
            try
            {
                var completeMultipartUploadRequest = new CompleteMultipartUploadRequest(BucketName, serviceRelaFileName, uploadId);
                foreach (var partETag in partETags)
                {
                    completeMultipartUploadRequest.PartETags.Add(partETag);
                }
                var result = client.CompleteMultipartUpload(completeMultipartUploadRequest);
                //Common.App.LogUtil.RecordRunLog("complete multi part succeeded");
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("完成分片上传异常!", ex);
                return false;
            }
            #endregion

            return true;
        }

        /// <summary>
        /// 删除单个文件
        /// </summary>
        /// <param name="serviceRelaFileName">如:@"RISImages\DX\20181115\2221659\000100000010\1.2.840.113619.2.203.4.2147483647.1542300170.371687.oss"</param>
        /// <returns></returns>
        public bool DeleteFileFromOss(string serviceRelaFileName)
        {
            try
            {
                serviceRelaFileName = serviceRelaFileName.Replace('\\', '/');
                client.DeleteObject(BucketName, serviceRelaFileName);
                return true;
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("从OSS上删除文件[" + serviceRelaFileName + "]异常!", ex);
                return false;
            }
        }

        /// <summary>
        /// 获取指定文件路径下的所有文件信息
        /// </summary>
        /// <param name="prefix">如：abc/def/hhh</param>
        /// <returns></returns>
        public List<OSSFileInfo> ListFilesFromOss(string prefix)
        {

            try
            {
                List<OSSFileInfo> files = new List<OSSFileInfo>();
                prefix = prefix.Replace('\\', '/');
                ObjectListing result = null;
                var listObjectsRequest = new ListObjectsRequest(BucketName)
                {
                    MaxKeys = 100,
                    Prefix = prefix,
                };
                do
                {
                    result = client.ListObjects(listObjectsRequest);
                    foreach (var summary in result.ObjectSummaries)
                    {
                        DateTime lastModifiedDate = new DateTime(1900, 1, 1);
                        if (summary.LastModified.Kind == DateTimeKind.Utc)
                        {
                            lastModifiedDate = TimeZoneInfo.ConvertTimeFromUtc(summary.LastModified, TimeZoneInfo.Local);
                        }
                        else
                        {
                            lastModifiedDate = summary.LastModified;
                        }
                        files.Add(new OSSFileInfo()
                        {
                            BucketName = summary.BucketName,
                            LastModifiedDate = lastModifiedDate,
                            RelaFileFullName = summary.Key,
                            Size = summary.Size,
                            StorageClass = summary.StorageClass,
                        });
                    }
                    listObjectsRequest.Marker = result.NextMarker;
                } while (result.IsTruncated);
                return files;
                //Console.WriteLine("List objects of bucket:{0} succeeded ", bucketName);
            }
            catch (OssException ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("从OSS上获取[" + prefix + "]下的文件列表异常!", ex);
                return null;
                //Console.WriteLine("Failed with error code: {0}; Error info: {1}. \nRequestID:{2}\tHostID:{3}", ex.ErrorCode, ex.Message, ex.RequestId, ex.HostId);
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("从OSS上获取[" + prefix + "]下的文件列表异常!", ex);
                return null;
                //Console.WriteLine("Failed with error info: {0}", ex.Message);
            }
        }

        public string UploadFileAndGetFileURL(string serviceRelaFileName, string localFileFullName, int Days = 7)
        {
            try
            {
                // 上传
                client.PutObject(BucketName, serviceRelaFileName, localFileFullName);

                OssClient requestClient = externalClient != null ? externalClient : client;

                var request = new GeneratePresignedUriRequest(BucketName, serviceRelaFileName, SignHttpMethod.Get);
                request.Expiration = DateTime.Now.AddDays(Days);
                var signedUrl = requestClient.GeneratePresignedUri(request);
                return signedUrl.ToString();
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("获取访问地址异常!", ex);
                return string.Empty;
            }
        }

    }
}
