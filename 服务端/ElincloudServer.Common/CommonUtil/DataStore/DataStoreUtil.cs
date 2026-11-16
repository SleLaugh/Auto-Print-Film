using ElincloudServer.Entity;

namespace ElincloudServer.Common
{
    public class DataStoreUtil
    {
        /// <summary>
        /// 获取对像存储实例
        /// </summary>
        /// <param name="provider"></param>
        /// <param name="accessKeyID"></param>
        /// <param name="accessKeySecret"></param>
        /// <param name="endPoint"></param>
        /// <param name="bucketName"></param>
        /// <returns></returns>
        /// <exception cref="ServiceException"></exception>
        public IOSS GetDataStoreIntance(OSSTypes provider, string accessKeyID, string accessKeySecret, string endPoint, string bucketName)
        {
            try
            {
                if (provider == OSSTypes.OBS)
                {
                    return new OBSUtil(accessKeyID, accessKeySecret, endPoint, bucketName);
                }
                else if (provider == OSSTypes.OSS)
                {
                    return new OSSUtil(accessKeyID, accessKeySecret, endPoint, bucketName);
                }
                else if (provider == OSSTypes.OOS)
                {
                    return new AmazonS3Util(accessKeyID, accessKeySecret, endPoint, bucketName);
                }
                else if (provider == OSSTypes.AmazonS3)
                {
                    return new AmazonS3Util(accessKeyID, accessKeySecret, endPoint, bucketName);
                }
                else
                {
                    throw new ServiceException("未知的对象存储服务商！");
                }
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("初始化OSS操作类失败!", ex);
                return null;
            }

        }

        /// <summary>
        /// 获取对像存储实例
        /// </summary>
        /// <param name="provider"></param>
        /// <param name="accessKeyID"></param>
        /// <param name="accessKeySecret"></param>
        /// <param name="endPoint"></param>
        /// <param name="externalEndPoint"></param>
        /// <param name="bucketName"></param>
        /// <returns></returns>
        /// <exception cref="ServiceException"></exception>
        public IOSS GetDataStoreIntance(OSSTypes provider, string accessKeyID, string accessKeySecret, string endPoint, string externalEndPoint, string bucketName)
        {
            try
            {
                if (provider == OSSTypes.OBS)
                {
                    return new OBSUtil(accessKeyID, accessKeySecret, endPoint, externalEndPoint, bucketName);
                }
                else if (provider == OSSTypes.OSS)
                {
                    return new OSSUtil(accessKeyID, accessKeySecret, endPoint, externalEndPoint, bucketName);
                }
                else if (provider == OSSTypes.OOS)
                {
                    return new AmazonS3Util(accessKeyID, accessKeySecret, endPoint, externalEndPoint, bucketName);
                }
                else if (provider == OSSTypes.AmazonS3)
                {
                    return new AmazonS3Util(accessKeyID, accessKeySecret, endPoint, externalEndPoint, bucketName);
                }
                else
                {
                    throw new ServiceException("未知的对象存储服务商！");
                }
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("初始化OSS操作类失败!", ex);
                return null;
            }

        }
    }
}
