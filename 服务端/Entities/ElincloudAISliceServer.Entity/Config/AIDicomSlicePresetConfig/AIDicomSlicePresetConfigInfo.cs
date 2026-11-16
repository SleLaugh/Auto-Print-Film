namespace ElincloudAISliceServer.Entity
{
    /// <summary>
    /// 切割影像预设配置
    /// </summary>
    public class AIDicomSlicePresetConfigInfo
    {
        /// <summary>
        /// 医院编号
        /// </summary>
        public string HospitalCode { get; set; }

        /// <summary>
        /// 窗宽窗位列表
        /// </summary>
        public List<WindowPreset> WindowPresets { get; set; } = new List<WindowPreset>();

        /// <summary>
        /// 序列检查部位tag值和检查部位的对照关系
        /// </summary>
        public List<BodyPartMapping> BodyPartMappings { get; set; } = new List<BodyPartMapping>();


        /// <summary>
        /// 设置初始值：只有为空时才写入默认切割规则，已有配置则原样保留。
        /// </summary>
        public void Initialize()
        {
            if (WindowPresets == null || WindowPresets.Count == 0)
            {
                WindowPresets = new List<WindowPreset>()
                {
                    new WindowPreset("脑窗", 80, 40),
                    new WindowPreset("骨窗", 2500, 400),
                    new WindowPreset("肺窗", 1600, -600),
                    new WindowPreset("纵隔窗", 350, 40),
                    new WindowPreset("软组织窗", 400, 50),
                    new WindowPreset("腹部窗", 350, 50),
                    new WindowPreset("肝窗", 150, 60),
                    new WindowPreset("肾窗", 350, 40),
                    new WindowPreset("胰腺窗", 300, 40),
                    new WindowPreset("骨盆窗", 400, 40),
                    new WindowPreset("脊柱窗", 2000, 300)
                };
            }


            if (BodyPartMappings == null || BodyPartMappings.Count == 0)
            {
                BodyPartMappings = new List<BodyPartMapping>()
                {
                    new BodyPartMapping("CHEST",     "肺"),
                    new BodyPartMapping("NECK",      "颈椎"),
                    new BodyPartMapping("HEAD",      "颅脑"),
                    new BodyPartMapping("BRAIN",     "颅脑"),
                    new BodyPartMapping("LSPINE",    "椎"),
                    new BodyPartMapping("SPINE",     "椎"),
                    new BodyPartMapping("PELVIS",    "骨盆"),
                    new BodyPartMapping("HIP",       "骨盆"),
                    new BodyPartMapping("EXTREMITY", "骨盆"),
                    new BodyPartMapping("CSPINE",    "颈椎")
                };
            }
        }
    }
}
