using Microsoft.AspNetCore.Mvc.Filters;

namespace ElincloudServer
{
    public class ResourcesFilter : Attribute, IResourceFilter
    {
        private static Dictionary<PathString, int> cacheConfigs = new Dictionary<PathString, int>();

        internal static void Init()
        {
            //cacheConfigs.Add(new PathString("/IM/Login"), 5);//登录接口在第一次调成功以后5分钟内直接返回缓存结果，而不去查询数据库
            //...其它缓存依需要按上面的方法进行配置，这里的配置信息可以考虑在数据库中进行设置
        }

        //用于存放缓存数据的字典集合
        private Dictionary<PathString, ResoucecCache> CachedResponses = new Dictionary<PathString, ResoucecCache>();

        //对请求进行处理
        public void OnResourceExecuting(ResourceExecutingContext context)
        {
            //获得请求的Path
            PathString key = context.HttpContext.Request.Path;
            //检测Path是否在缓存集合中
            if (cacheConfigs.ContainsKey(key))
            {
                if (CachedResponses.ContainsKey(key))
                {
                    if (DateTime.Now <= CachedResponses[key].CacheDate.AddMinutes(cacheConfigs[key]))//缓存在有效期内
                    {
                        //存在缓存，直接将缓存好的结果做返回
                        context.Result = CachedResponses[key].Result;
                    }
                    else
                    {
                        //缓存过期，直接删除接口在执行完成后重新缓存
                        CachedResponses.Remove(key);
                    }
                }
            }

        }

        public void OnResourceExecuted(ResourceExecutedContext context)
        {
            PathString key = context.HttpContext.Request.Path;
            if (cacheConfigs.ContainsKey(key))
            {
                if (CachedResponses.ContainsKey(key))
                {
                    CachedResponses[key].CacheDate = DateTime.Now;
                    CachedResponses[key].Result = context.Result;
                }
                else
                {
                    CachedResponses.Add(key, new ResoucecCache()
                    {
                        Method = key,
                        CacheDate = DateTime.Now,
                        Result = context.Result
                    });
                }
            }
        }


    }
}
