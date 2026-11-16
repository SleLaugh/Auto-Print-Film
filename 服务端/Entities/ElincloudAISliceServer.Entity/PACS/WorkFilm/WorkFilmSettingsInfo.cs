using ElincloudServer.Entity;

namespace ElincloudAISliceServer.Entity
{
    public class WorkFilmSettingsInfo
    {
        public GeneralInfo CommonInfo { get; set; } = new GeneralInfo();

        public List<WindowInfo> WindowInfo { get; set; } = new List<WindowInfo>();

        public List<PrinterInfo> PrinterInfo { get; set; } = new  List<PrinterInfo>();

        public List<FilmSizeInfo> FilmSizeInfo = new List<FilmSizeInfo>();
    }
}
