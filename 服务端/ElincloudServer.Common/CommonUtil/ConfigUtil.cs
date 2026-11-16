using ElincloudServer.Entity.Config.DataStore;
using Microsoft.Extensions.Configuration;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace ElincloudServer.Common
{
    public class ConfigUtil
    {
        private ConfigurationManager configurationManager = null;
        string configFileFullName = AppDomain.CurrentDomain.BaseDirectory + "appsettings.json";

        public ConfigUtil()
        {
            if (!File.Exists(configFileFullName))
            {
                File.Create(configFileFullName);
            }
            configurationManager = new ConfigurationManager();
            configurationManager.AddJsonFile("appsettings.json", optional: true, reloadOnChange: true);
        }

        #region 数据库配置

        private string _dbSection = "DBConnectionInfo";
        private string _IP = "IP";
        private string _Port = "Port";
        private string _UserName = "UserName";
        private string _Password = "Password";
        private string _DBType = "DBType";


        /// <summary>
        /// IP
        /// </summary>
        public string IP
        {
            get
            {
                return getValue(_dbSection, _IP);
            }
            set
            {
                setValue(_dbSection, _IP, value);
            }
        }

        /// <summary>
        /// 端口
        /// </summary>
        public string Port
        {
            get
            {
                return getValue(_dbSection, _Port);
            }
            set
            {
                setValue(_dbSection, _Port, value);
            }
        }

        /// <summary>
        /// 用户名
        /// </summary>
        public string UserName
        {
            get
            {
                return getValue(_dbSection, _UserName);
            }
            set
            {
                setValue(_dbSection, _UserName, value);
            }
        }

        /// <summary>
        /// 密码
        /// </summary>
        public string Password
        {
            get
            {
                return ElincloudServer.Common.Utils.StringUtil.Decrypt(getValue(_dbSection, _Password));
            }
            set
            {
                setValue(_dbSection, _Password, ElincloudServer.Common.Utils.StringUtil.Encrypt(value));
            }
        }

        /// <summary>
        /// 数据库类型
        /// </summary>
        public string DBType
        {
            get
            {
                return getValue(_dbSection, _DBType);
            }
            set
            {
                setValue(_dbSection, _DBType, value);
            }
        }

        private string isElincloudServerLog = "IsElincloudServerLog";

        public bool IsElincloudServerLog
        {
            get
            {
                string ret = getValue(isElincloudServerLog);
                return ret.ToLower() == "true";
            }
            set { setValue(isElincloudServerLog, value.ToString()); }
        }

        #endregion

        #region 服务运行地址

        private string _ServerUrl = "ServerUrl";
        /// <summary>
        /// IP
        /// </summary>
        public string ServerUrl
        {
            get
            {
                return getValue(_ServerUrl);
            }
            set
            {
                setValue(_ServerUrl, value);
            }
        }

        #endregion

        #region 文件服务运行地址

        private string _FileWebServiceUrl = "FileWebServiceUrl";
        /// <summary>
        /// IP
        /// </summary>
        public string FileWebServiceUrl
        {
            get
            {
                return getValue(_FileWebServiceUrl);
            }
            set
            {
                setValue(_FileWebServiceUrl, value);
            }
        }
        #endregion

        #region 文件存储地址

        private string _FileStoragePath = "FileStoragePath";
        /// <summary>
        /// FileStoragePath
        /// </summary>
        public string FileStoragePath
        {
            get
            {
                return getValue(_FileStoragePath);
            }
            set
            {
                setValue(_FileStoragePath, value);
            }
        }
        #endregion

        #region Cookie作用域

        private string _CookieDomain = "CookieDomain";
        /// <summary>
        /// CookieDomain
        /// </summary>
        public string CookieDomain
        {
            get
            {
                return getValue(_CookieDomain);
            }
            set
            {
                setValue(_CookieDomain, value);
            }
        }

        #endregion

        #region 版本号

        private string _Version = "Version";
        /// <summary>
        /// IP
        /// </summary>
        public string Version
        {
            get
            {
                return getValue(_Version);
            }
            set
            {
                setValue(_Version, value);
            }
        }

        #endregion

        #region 插件版本号
        private string _Plugin = "Plugin";
        private string _PluginVersion = "Version";
        private string _PluginRelativePath = "RelativePath";
        /// <summary>
        /// 版本
        /// </summary>
        public string PluginVersion
        {
            get
            {
                return getValue(_Plugin, _PluginVersion);
            }
            set
            {
                setValue(_Plugin, _PluginVersion, value);
            }
        }

        /// <summary>
        /// 相对下载路径
        /// </summary>
        public string PluginRelativePath
        {
            get
            {
                return getValue(_Plugin, _PluginRelativePath);
            }
            set
            {
                setValue(_Plugin, _PluginRelativePath, value);
            }
        }

        #endregion

        #region 程序运行模式

        private string _IsDebug = "IsDebug";
        /// <summary>
        /// IP
        /// </summary>
        public bool IsDebug
        {
            get
            {
                string ret = getValue(_IsDebug);
                return ret.ToLower() == "true";
            }
            set
            {
                setValue(_IsDebug, value.ToString());
            }
        }

        #endregion

        #region 程序运行环境

        /// <summary>
        /// IsWindow
        /// </summary>
        public bool IsWindow
        {
            get
            {
                return Environment.OSVersion.Platform < PlatformID.Unix;
            }
        }

        #endregion

        #region 公共方法

        protected string getValue(string key)
        {
            try
            {
                var value = configurationManager.GetSection(key).Value;
                return value == null ? "" : value;
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("配置文件读取异常！", ex);
                return string.Empty;
            }
        }

        protected void setValue(string key, string value)
        {
            try
            {
                JObject jsonObject = null;
                using (StreamReader file = new StreamReader(configFileFullName))
                {
                    using (JsonTextReader reader = new JsonTextReader(file))
                    {
                        jsonObject = (JObject)JToken.ReadFrom(reader);
                        jsonObject[key] = value;
                    }
                }
                using (var writer = new StreamWriter(configFileFullName))
                {
                    using (JsonTextWriter jsonwriter = new JsonTextWriter(writer))
                    {
                        jsonObject.WriteTo(jsonwriter);
                    }
                }
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("配置文件写入异常！", ex);
            }

        }

        protected string getValue(string setion, string key)
        {
            try
            {
                var value = configurationManager.GetSection(string.Format("{0}:{1}", setion, key)).Value;
                return value == null ? "" : value;
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("配置文件读取异常！", ex);
                return string.Empty;
            }
        }
        protected void setValue(string setion, string key, string value)
        {
            try
            {
                JObject jsonObject = null;
                using (StreamReader file = new StreamReader(configFileFullName))
                {
                    using (JsonTextReader reader = new JsonTextReader(file))
                    {
                        jsonObject = (JObject)JToken.ReadFrom(reader);
                        var s = jsonObject[setion];
                        if (s != null)
                        {
                            s[key] = value;
                        }
                        else
                        {
                            jsonObject[setion] = new JObject { { key, value } };
                        }
                    }
                }
                using (var writer = new StreamWriter(configFileFullName))
                {
                    using (JsonTextWriter jsonwriter = new JsonTextWriter(writer))
                    {
                        jsonObject.WriteTo(jsonwriter);
                    }
                }
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("配置文件写入异常！", ex);
            }

        }

        #endregion

        #region 部署方式

        private string _deployment = "Deployment";
        /// <summary>
        /// 部署方式
        /// </summary>
        public string Deployment
        {
            get
            {
                return getValue(_deployment);
            }
            set
            {
                setValue(_deployment, value);
            }
        }

        #endregion

        #region 归档进程数


        private string _ArchiveProssessCount = "ArchiveProssessCount";
        /// <summary>
        /// 归档进程数
        /// </summary>
        public string ArchiveProssessCount
        {
            get
            {
                return getValue(_ArchiveProssessCount);
            }
            set
            {
                setValue(_ArchiveProssessCount, value);
            }
        }

        #endregion

        #region 远程仓库存储服务地址和存储文件列表配置

        private void setValue(string key, JToken value)
        {
            try
            {
                JObject jsonObject = null;

                using (StreamReader file = new StreamReader(configFileFullName))
                using (JsonTextReader reader = new JsonTextReader(file))
                {
                    jsonObject = (JObject)JToken.ReadFrom(reader);
                    jsonObject[key] = value; // 这里直接存 JToken → 数组/对象都支持
                }

                using (var writer = new StreamWriter(configFileFullName))
                using (JsonTextWriter jsonwriter = new JsonTextWriter(writer)
                {
                    Formatting = Formatting.Indented // 自动格式化，好看
                })
                {
                    jsonObject.WriteTo(jsonwriter);
                }
            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("配置文件写入异常！", ex);
            }
        }
        private JToken getJToken(string key)
        {
            try
            {
                var section = configurationManager.GetSection(key);
                if (!section.Exists())
                    return new JArray();

                var children = section.GetChildren().ToList();
                JArray array = new JArray();

                foreach (var child in children)
                {
                    JObject obj = new JObject();
                    foreach (var prop in child.GetChildren())
                    {
                        obj[prop.Key] = prop.Value;
                    }
                    array.Add(obj);
                }

                return array;
            }
            catch
            {
                return new JArray();
            }
        }
        private string _RemoteDepotStoreService = "RemoteDepotStoreServiceUrl";
        private string _FolderConfigInfos = "FolderConfigInfos";

        /// <summary>
        /// 远程仓库存储服务端的服务地址
        /// </summary>
        public string RemoteDepotStoreServiceUrl
        {
            get
            {
                return getValue(_RemoteDepotStoreService);
            }
            set
            {
                setValue(_RemoteDepotStoreService, value);
            }
        }

        /// <summary>
        /// 文件夹配置列表
        /// </summary>
        public JArray FolderConfigInfos
        {
            get
            {
                try
                {
                    // 直接获取 JSON 数组
                    JToken jToken = getJToken(_FolderConfigInfos);
                    return getJToken(_FolderConfigInfos) as JArray ?? new JArray();
                }
                catch (Exception ex)
                {
                    ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("FolderConfigInfos 读取异常！", ex);
                    return new JArray();
                }
            }
            set
            {
                try
                {
                    setValue(_FolderConfigInfos, value);
                }
                catch (Exception ex)
                {
                    ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("FolderConfigInfos 写入异常！", ex);
                }
            }
        }
        #endregion
    }
}
