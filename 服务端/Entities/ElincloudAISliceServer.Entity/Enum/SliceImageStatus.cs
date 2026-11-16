namespace ElincloudAISliceServer.Entity
{
    /// <summary>
    /// 影像切割状态
    /// </summary>
    public enum SliceImageStatus
    {
        未切割 = 0,
        正在切割 = 1,
        已切割 = 2,
        不切割 = 3,
        切割失败 = 4
    }
}
