using ElincloudServer.Entity;
using MySql.Data.MySqlClient;
using Oracle.ManagedDataAccess.Client;
using Microsoft.Data.SqlClient;
using Dm;

namespace ElincloudServer.Common
{
    public class DataBaseUtil
    {
        public  bool TestConnection(DBTypes dbType, string DBIP, string DBPort, string DBUserName, string DBPassword, string dBName = "master")
        {
            try
            {
                if (dbType == DBTypes.SqlServer)
                {


                    //sqlserver 当ip是以  192.168.31.123\MSSQL 这种带实例的方式访问的时候，连接字符串的DataSource后面是不需要跟端口的，跟了端口会导致连接失败
                    string dataSource = DBIP.Contains("\\") ? DBIP : string.Format("{0},{1}", DBIP, DBPort);
                    string connectionStr = string.Format("Data Source={0};Initial Catalog={1};User ID={2};Password={3}; TrustServerCertificate = true", dataSource, dBName, DBUserName, DBPassword);
                    using (SqlConnection conn = new SqlConnection(connectionStr))
                    {
                        try
                        {
                            conn.Open();
                        }
                        catch (Exception ex)
                        {
                            ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("测试数据库连接失败！", ex);
                            return false;
                        }
                    }
                }
                else if (dbType == DBTypes.Oracle)
                {
                    string connectionStr = string.Format("Data Source=(DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST={0})(PORT={1}))(CONNECT_DATA=(SERVICE_NAME={2})));Persist Security Info=True;User ID={3};Password={4};", DBIP, DBPort, dBName, DBUserName, DBPassword);
                    using (OracleConnection connection = new OracleConnection(connectionStr))
                    {
                        try
                        {
                            connection.Open();
                        }
                        catch (Exception ex)
                        {
                            ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("测试数据库连接失败！", ex);
                            return false;
                        }
                    }
                }
                else if (dbType == DBTypes.MySql)
                {
                    dBName = "mysql";
                    string connectionStr = string.Format("Data Source={0};user id={1}; password={2}; database={3}; pooling=false;charset=utf8mb4;Allow Zero Datetime=True;port={4}", DBIP, DBUserName, DBPassword, dBName, DBPort);
                    using (MySqlConnection connection = new MySqlConnection(connectionStr))
                    {
                        try
                        {
                            connection.Open();
                        }
                        catch (Exception ex)
                        {
                            ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("测试数据库连接失败！", ex);
                            return false;
                        }
                    }
                }
                else if (dbType == DBTypes.DM)
                {
                    string connectionStr = string.Format("Server={0};Port={1};Database={2};User ID={3};Password={4};", DBIP, DBPort, dBName, DBUserName, DBPassword);
                    using (DmConnection connection = new DmConnection(connectionStr))
                    {
                        try
                        {
                            connection.Open();
                        }
                        catch (Exception ex)
                        {
                            ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("测试数据库连接失败！", ex);
                            return false;
                        }
                    }
                }
                return true;

            }
            catch (Exception ex)
            {
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("测试数据库连接失败！", ex);
                return false;
            }
        }
    }
}
