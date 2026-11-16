namespace ElincloudServer.BLServices
{
    public class BLService
    {
        private static SystemBLService.SystemBL systemBLService = null;
        public static SystemBLService.SystemBL SystemBLService
        {
            get
            {
                if (systemBLService == null)
                {
                    systemBLService = new SystemBLService.SystemBL();
                }
                return systemBLService;
            }
        }

        private static LogBLService.LogBL logBLService = null;
        public static LogBLService.LogBL LogBLService
        {
            get
            {
                if (logBLService == null)
                {
                    logBLService = new LogBLService.LogBL();
                }
                return logBLService;
            }
        }

        private static ConfigBLService.ConfigBL configBLService = null;
        public static ConfigBLService.ConfigBL ConfigBLService
        {
            get
            {
                if (configBLService == null)
                {
                    configBLService = new ConfigBLService.ConfigBL();
                }
                return configBLService;
            }
        }

        private static ElincloudAISliceServer.AISliceBLService.AISliceBL aiSliceBLService = null;
        public static ElincloudAISliceServer.AISliceBLService.AISliceBL AISliceBLService
        {
            get
            {
                if (aiSliceBLService == null)
                {
                    aiSliceBLService = new ElincloudAISliceServer.AISliceBLService.AISliceBL();
                }
                return aiSliceBLService;
            }
        }
    }
}
