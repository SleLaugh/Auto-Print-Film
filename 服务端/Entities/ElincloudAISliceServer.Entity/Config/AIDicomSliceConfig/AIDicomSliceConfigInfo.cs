namespace ElincloudAISliceServer.Entity
{
    /// <summary>
    /// 切割影像配置
    /// </summary>
    public class AIDicomSliceConfigInfo
    {
        /// <summary>
        /// 智能影像切割可执行程序路径
        /// </summary>
        public string ExecutablePath { get; set; } = string.Empty;

        /// <summary>
        /// 最大并发数量
        /// </summary>
        public int MaxConcurrentCount { get; set; } = 3;

        /// <summary>
        /// 切割规则
        /// </summary>
        public List<SliceRuleConfigInfo> SliceRuleConfigInfos { get; set; } = new List<SliceRuleConfigInfo>();

        /// <summary>
        /// 设置初始值：只有为空时才写入默认切割规则，已有配置则原样保留。
        /// </summary>
        public void Initialize()
        {
            if (SliceRuleConfigInfos == null || SliceRuleConfigInfos.Count == 0)
            {
                SliceRuleConfigInfos = new List<SliceRuleConfigInfo>()
                {
                    new SliceRuleConfigInfo("CT", "颅脑", PlaneEnums.横状位, SliceTypes.MPR, new List<SliceRegions>() { SliceRegions.Default }),
                    new SliceRuleConfigInfo("CT", "颅脑", PlaneEnums.冠状位, SliceTypes.MPR, new List<SliceRegions>() { SliceRegions.Default }),
                    new SliceRuleConfigInfo("CT", "颅脑", PlaneEnums.矢状位, SliceTypes.MPR, new List<SliceRegions>() { SliceRegions.Default }),

                    new SliceRuleConfigInfo("CT", "颈椎", PlaneEnums.横状位, SliceTypes.MPR, new List<SliceRegions>() { SliceRegions.Default }),
                    new SliceRuleConfigInfo("CT", "颈椎", PlaneEnums.横状位, SliceTypes.椎间隙, new List<SliceRegions>() { SliceRegions.C2_C3, SliceRegions.C3_C4, SliceRegions.C4_C5, SliceRegions.C5_C6, SliceRegions.C6_C7 }),
                    new SliceRuleConfigInfo("CT", "颈椎", PlaneEnums.冠状位, SliceTypes.MPR, new List<SliceRegions>() { SliceRegions.Default }),
                    new SliceRuleConfigInfo("CT", "颈椎", PlaneEnums.冠状位, SliceTypes.CPR, new List<SliceRegions>() { SliceRegions.Default }),
                    new SliceRuleConfigInfo("CT", "颈椎", PlaneEnums.矢状位, SliceTypes.MPR, new List<SliceRegions>() { SliceRegions.Default }),
                    new SliceRuleConfigInfo("CT", "颈椎", PlaneEnums.矢状位, SliceTypes.CPR, new List<SliceRegions>() { SliceRegions.Default }),

                    new SliceRuleConfigInfo("CT", "胸椎", PlaneEnums.横状位, SliceTypes.MPR, new List<SliceRegions>() { SliceRegions.Default }),
                    new SliceRuleConfigInfo("CT", "胸椎", PlaneEnums.冠状位, SliceTypes.MPR, new List<SliceRegions>() { SliceRegions.Default }),
                    new SliceRuleConfigInfo("CT", "胸椎", PlaneEnums.冠状位, SliceTypes.CPR, new List<SliceRegions>() { SliceRegions.Default }),
                    new SliceRuleConfigInfo("CT", "胸椎", PlaneEnums.矢状位, SliceTypes.MPR, new List<SliceRegions>() { SliceRegions.Default }),
                    new SliceRuleConfigInfo("CT", "胸椎", PlaneEnums.矢状位, SliceTypes.CPR, new List<SliceRegions>() { SliceRegions.Default }),

                    new SliceRuleConfigInfo("CT", "胸腰椎", PlaneEnums.横状位, SliceTypes.MPR, new List<SliceRegions>() { SliceRegions.Default }),
                    new SliceRuleConfigInfo("CT", "胸腰椎", PlaneEnums.冠状位, SliceTypes.MPR, new List<SliceRegions>() { SliceRegions.Default }),
                    new SliceRuleConfigInfo("CT", "胸腰椎", PlaneEnums.冠状位, SliceTypes.CPR, new List<SliceRegions>() { SliceRegions.Default }),
                    new SliceRuleConfigInfo("CT", "胸腰椎", PlaneEnums.矢状位, SliceTypes.MPR, new List<SliceRegions>() { SliceRegions.Default }),
                    new SliceRuleConfigInfo("CT", "胸腰椎", PlaneEnums.矢状位, SliceTypes.CPR, new List<SliceRegions>() { SliceRegions.Default }),

                    new SliceRuleConfigInfo("CT", "全脊柱", PlaneEnums.横状位, SliceTypes.MPR, new List<SliceRegions>() { SliceRegions.Default }),
                    new SliceRuleConfigInfo("CT", "全脊柱", PlaneEnums.冠状位, SliceTypes.MPR, new List<SliceRegions>() { SliceRegions.Default }),
                    new SliceRuleConfigInfo("CT", "全脊柱", PlaneEnums.冠状位, SliceTypes.CPR, new List<SliceRegions>() { SliceRegions.Default }),
                    new SliceRuleConfigInfo("CT", "全脊柱", PlaneEnums.矢状位, SliceTypes.MPR, new List<SliceRegions>() { SliceRegions.Default }),
                    new SliceRuleConfigInfo("CT", "全脊柱", PlaneEnums.矢状位, SliceTypes.CPR, new List<SliceRegions>() { SliceRegions.Default }),

                    new SliceRuleConfigInfo("CT", "腰椎", PlaneEnums.横状位, SliceTypes.MPR, new List<SliceRegions>() { SliceRegions.Default }),
                    new SliceRuleConfigInfo("CT", "腰椎", PlaneEnums.横状位, SliceTypes.椎间隙, new List<SliceRegions>() { SliceRegions.L1_L2, SliceRegions.L2_L3, SliceRegions.L3_L4, SliceRegions.L4_L5, SliceRegions.L5_S1 }),
                    new SliceRuleConfigInfo("CT", "腰椎", PlaneEnums.冠状位, SliceTypes.MPR, new List<SliceRegions>() { SliceRegions.Default }),
                    new SliceRuleConfigInfo("CT", "腰椎", PlaneEnums.冠状位, SliceTypes.CPR, new List<SliceRegions>() { SliceRegions.Default }),
                    new SliceRuleConfigInfo("CT", "腰椎", PlaneEnums.矢状位, SliceTypes.MPR, new List<SliceRegions>() { SliceRegions.Default }),
                    new SliceRuleConfigInfo("CT", "腰椎", PlaneEnums.矢状位, SliceTypes.CPR, new List<SliceRegions>() { SliceRegions.Default }),

                    new SliceRuleConfigInfo("CT", "肺部", PlaneEnums.横状位, SliceTypes.MPR, new List<SliceRegions>() { SliceRegions.Default }),
                    new SliceRuleConfigInfo("CT", "肺部", PlaneEnums.冠状位, SliceTypes.MPR, new List<SliceRegions>() { SliceRegions.Default }),
                    new SliceRuleConfigInfo("CT", "肺部", PlaneEnums.矢状位, SliceTypes.MPR, new List<SliceRegions>() { SliceRegions.Default }),

                    new SliceRuleConfigInfo("CT", "髋骨", PlaneEnums.横状位, SliceTypes.MPR, new List<SliceRegions>() { SliceRegions.Default }),
                    new SliceRuleConfigInfo("CT", "髋骨", PlaneEnums.冠状位, SliceTypes.MPR, new List<SliceRegions>() { SliceRegions.Default }),
                    new SliceRuleConfigInfo("CT", "髋骨", PlaneEnums.矢状位, SliceTypes.MPR, new List<SliceRegions>() { SliceRegions.Default })
                };
            }
        }
    }
}
