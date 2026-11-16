
using ElincloudServer.Entity;
using OBS;
using OBS.Model;

namespace ElincloudServer.Common
{
    public class OBSUtil : IOSS
    {
        private string _bucketName = string.Empty;
        public string BucketName
        {
            get { return _bucketName; }
        }


        ObsClient client = null;

        /// <summary>
        /// 外网链接对象
        /// </summary>
        ObsClient externalClient = null;

        //public OBSUtil()
        //{
        //    string accessKeyID = Common.App.RegUtil.GetOSSAccessKeyID();
        //    string accessKeySecret = Common.App.RegUtil.GetOSSAccessKeySecret();
        //    string endPoint = Common.App.RegUtil.GetOSSEndPoint();
        //    _bucketName = Common.App.RegUtil.GetOSSBucketName();
        //    client = new ObsClient(accessKeyID, accessKeySecret, endPoint);
        //}

        public OBSUtil(string keyID, string keySecret, string endpoint, string bucketName)
        {
            _bucketName = bucketName;
            client = new ObsClient(keyID, keySecret, endpoint);
        }

        public OBSUtil(string keyID, string keySecret, string endpoint, string externalEndPoint, string bucketName)
        {
            _bucketName = bucketName;
            client = new ObsClient(keyID, keySecret, endpoint);

            if (!string.IsNullOrEmpty(externalEndPoint))
            {
                externalClient = new ObsClient(keyID, keySecret, externalEndPoint);
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
                GetObjectMetadataResponse metadata = client.GetObjectMetadata(BucketName, serviceRelaFileName);
                //枚举常量和说明 COLD 归档存储, STANDARD 标准存储 ,WARM 低频访问存储
                if (metadata.StorageClass == StorageClassEnum.Cold)
                {
                    RestoreObjectRequest request = new RestoreObjectRequest()
                    {
                        BucketName = BucketName,
                        ObjectKey = serviceRelaFileName,
                        Days = 5,
                        Tier = RestoreTierEnum.Expedited,//快速取回1-5分钟完成
                        VersionId = ""
                    };
                    try
                    {
                        RestoreObjectResponse result = client.RestoreObject(request);
                        if (result.StatusCode == System.Net.HttpStatusCode.OK)
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
            GetObjectRequest request = new GetObjectRequest()
            {
                BucketName = BucketName,
                ObjectKey = serviceRelaFileName,
            };
            GetObjectResponse result = client.GetObject(request);
            string localFileName = Utils.StringUtil.GetFileName(serviceRelaFileName);
            string tempFolder = isDecompress?ElincloudServer.Common.Utils.StringUtil.Combine(localFolder, "temp"): localFolder;//不解压缩时直接到目标目录
            string localFileFullName = ElincloudServer.Common.Utils.StringUtil.Combine(tempFolder, localFileName);
            using (var requestStream = result.OutputStream)
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
                    //System.IO.File.Delete(ret.ZipFile);
                    System.IO.Directory.Delete(tempFolder, true);
                }
            }
        }

        /// <summary>
        /// 加密压缩文件夹并以oss为后缀上传
        /// </summary>
        /// <param name="serviceRelaFileName">如:@"RISImages\DX\20181115\2221659\000100000010\1.2.840.113619.2.203.4.2147483647.1542300170.371687.oss"</param>
        /// <param name="localFolder">如:@"E:\FTPFiles\RISImages\CT\20181016\00000100000006"</param>
        public long PutFolderToOss(string localFolder, string serviceRelaFileName, bool isEncrypt)
        {
            try
            {
                string fileName = Utils.StringUtil.GetFileName(serviceRelaFileName);

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
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("上传文件到OSS异常!", ex);
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
                InitiateMultipartUploadRequest initiateRequest = new InitiateMultipartUploadRequest
                {
                    BucketName = BucketName,
                    ObjectKey = serviceRelaFileName
                };
                InitiateMultipartUploadResponse result = client.InitiateMultipartUpload(initiateRequest);
                uploadId = result.UploadId;
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("初始化OSS分片上传存储空间失败!", ex);
                return false;
            }


            List<UploadPartResponse> uploadResponses = new List<UploadPartResponse>();
            // 2. 上传段
            try
            {
                string filePath = ossFileFullName;//上传的本地文件路径，需要指定到具体的文件名
                long contentLength = new FileInfo(filePath).Length;
                //long partSize = 5 * 1024 * 1024;
                long partSize = 5242880;//5M
                long filePosition = 0;
                for (int i = 1; filePosition < contentLength; i++)
                {
                    UploadPartRequest uploadRequest = new UploadPartRequest
                    {
                        BucketName = BucketName,
                        ObjectKey = serviceRelaFileName,
                        UploadId = uploadId,
                        PartNumber = i,
                        PartSize = partSize,
                        Offset = filePosition,
                        FilePath = filePath
                    };
                    uploadResponses.Add(client.UploadPart(uploadRequest));
                    filePosition += partSize;
                }
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("分片上传异常!", ex);
                //取消分段上传任务
                try
                {
                    AbortMultipartUploadRequest request = new AbortMultipartUploadRequest
                    {
                        BucketName = BucketName,
                        ObjectKey = serviceRelaFileName,
                        UploadId = uploadId,//待取消的分段上传任务的Id
                    };
                    AbortMultipartUploadResponse response = client.AbortMultipartUpload(request);
                }
                catch (ObsException e)
                {
                    ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("取消分段上传任务异常!", e);
                }
                return false;
            }


            // 3.合并段
            try
            {
                CompleteMultipartUploadRequest completeRequest = new CompleteMultipartUploadRequest()
                {
                    BucketName = BucketName,
                    ObjectKey = serviceRelaFileName,
                    UploadId = uploadId,
                };
                completeRequest.AddPartETags(uploadResponses);
                CompleteMultipartUploadResponse completeUploadResponse = client.CompleteMultipartUpload(completeRequest);
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
                DeleteObjectRequest request = new DeleteObjectRequest()
                {
                    BucketName = BucketName,
                    ObjectKey = serviceRelaFileName,
                    VersionId = ""
                };
                DeleteObjectResponse result = client.DeleteObject(request);
                if (result.OriginalResponse != null) result.OriginalResponse.Dispose();
                if (result.StatusCode == System.Net.HttpStatusCode.OK)
                {
                    return true;
                }
                else if (result.StatusCode == System.Net.HttpStatusCode.NoContent)
                {
                    //当发现服务端没有该文件的时候程序可以认为该文件删除成功，直接返回true
                    return true;
                }
                else
                {
                    ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("从OSS上删除文件[" + serviceRelaFileName + "]失败!StatusCode=" + result.StatusCode.ToString());
                    return false;
                }
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
                ListObjectsResponse result = null;
                var listObjectsRequest = new ListObjectsRequest()
                {
                    BucketName = BucketName,
                    MaxKeys = 100,
                    Prefix = prefix,
                };
                do
                {
                    result = client.ListObjects(listObjectsRequest);
                    foreach (var summary in result.ObsObjects)
                    {
                        DateTime lastModifiedDate = new DateTime(1900, 1, 1);
                        if (summary.LastModified != null)
                        {
                            if (summary.LastModified.Value.Kind == DateTimeKind.Utc)
                            {
                                lastModifiedDate = TimeZoneInfo.ConvertTimeFromUtc((DateTime)summary.LastModified, TimeZoneInfo.Local);
                            }
                            else
                            {
                                lastModifiedDate = (DateTime)summary.LastModified;
                            }
                        }
                        files.Add(new OSSFileInfo()
                        {
                            BucketName = BucketName,
                            LastModifiedDate = lastModifiedDate,
                            RelaFileFullName = summary.ObjectKey,
                            Size = summary.Size,
                            StorageClass = summary.StorageClass == null ? "" : summary.StorageClass.ToString(),
                        });
                    }
                    listObjectsRequest.Marker = result.NextMarker;
                } while (result.IsTruncated);
                return files;
                //Console.WriteLine("List objects of bucket:{0} succeeded ", bucketName);
            }
            catch (ObsException ex)
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
                PutObjectRequest putRequest = new PutObjectRequest()
                {
                    BucketName = BucketName,
                    ObjectKey = serviceRelaFileName,
                    FilePath = localFileFullName
                };
                client.PutObject(putRequest);

                ObsClient requestClient = externalClient != null ? externalClient : client;

                CreateTemporarySignatureRequest request = new CreateTemporarySignatureRequest();
                request.Method = HttpVerb.GET;
                request.BucketName = BucketName;
                request.ObjectKey = serviceRelaFileName;
                request.Expires = DateTime.Now.AddDays(Days).Ticks;

                CreateTemporarySignatureResponse response = requestClient.CreateTemporarySignature(request);
                return response.SignUrl.ToString();

            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("获取访问地址异常!", ex);
                return string.Empty;
            }
        }
    }
}
