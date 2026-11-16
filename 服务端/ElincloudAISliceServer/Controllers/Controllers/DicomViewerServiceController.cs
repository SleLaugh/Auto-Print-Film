using Dicom;
using Dicom.Imaging;
using Dicom.IO;
using Dicom.Network;
using Dicom.Printing;
using ElincloudServer.Common;
using ElincloudServer.CommonBLService;
using ElincloudServer.Entity;
using Microsoft.AspNetCore.Mvc;
using SkiaSharp;
using DicomClient = Dicom.Network.Client.DicomClient;
using FilmPrintRecordInfo = ElincloudAISliceServer.Entity.FilmPrintRecordInfo;

namespace ElincloudServer.Controllers
{
    public class DicomViewerServiceController : BaseController
    {
        #region 测试连接
        /// <summary>
        /// 测试连接
        /// </summary>
        /// <param name="PrinterInfo">打印机信息</param>
        /// <returns>string </returns>
        public JsonResult Echo(DicomServerInfo serverInfo)
        {
            return Excute(new Func<object>(() =>
            {
                // 定义返回对象
                ClientResult ret = new();

                try
                {
                    bool dicomReceived = false; // 定义连接标识
                                                // 创建连接
                    var client = new Dicom.Network.Client.DicomClient(serverInfo.IP, serverInfo.Port, false, "localAE", serverInfo.AE, 1000, 1000);
                    // 请求连接
                    client.AddRequestAsync(new DicomCEchoRequest
                    {
                        OnResponseReceived = (rq, rsp) =>
                        {
                            dicomReceived = true;
                            ret.Success = rsp.Status == DicomStatus.Success;
                            ret.Message = ret.Success ? "连接成功！" : "连接失败！";
                        },
                    }).ConfigureAwait(false);
                    // 发送连接
                    Task task = client.SendAsync();
                    // 等待连接2s
                    task.Wait(2000);
                    // 连接不成功 超时
                    if (!dicomReceived)
                    {
                        ret.Success = false;
                        ret.Message = "请求超时！";
                    }
                }
                catch (Exception ex)
                {
                    ret.Success = false;
                    ret.Message = "E00000448";
                    ret.Message += ex.InnerException == null ? ex.Message : ex.InnerException.Message;
                    ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("连接服务器异常！", ex);
                }
                return ret;
            }), "E00000107");
        }
        #endregion

        #region 上传文件

        /// <summary>
        /// 上传文件
        /// </summary>
        /// <param name="files"></param>
        /// <param name="relaFolderPath"></param>
        /// <returns></returns>
        public JsonResult UploadFiles(IFormFileCollection files, string relaFolderPath, bool isDecompress, bool isEncrypt = false)
        {
            return Excute(new Func<List<string>>(() =>
            {
                string result = string.Empty;
                try
                {
                    //这里跟前端调用或调用方 协商好，请求的content 类型必须为multipart/form-data
                    if (!HttpContext.Request.ContentType.Contains("multipart"))
                    {

                        result = "上传文件协议必须为multipart/form-data！";
                    }

                    if (files != null && files.Count > 0)
                    {
                        ServiceCommonConfigInfo commonConfigInfo = CommonBL.GetServiceCommonConfigInfo();

                        List<string> relaFileNames = new List<string>();
                        string folder = Common.Utils.StringUtil.Combine(commonConfigInfo.FileStoragePath, relaFolderPath);
                        if (!System.IO.Directory.Exists(folder))
                        {
                            Common.Utils.FileUtil.CreateDirectory(folder);
                        }
                        foreach (var file in files)
                        {
                            string fileName = Utils.StringUtil.GetFileName(file.FileName);
                            string filePath = Common.Utils.StringUtil.Combine(folder, Utils.StringUtil.GetFileName(file.FileName));
                            // 将文件保存到服务器
                            using (var stream = new FileStream(filePath, FileMode.Create))
                            {
                                file.CopyTo(stream);
                            }
                            if (isDecompress)
                            {
                                string compressFolderPath = Utils.StringUtil.GetDirectoryName(filePath);
                                Common.Utils.GZipUtil.Decompress(filePath, compressFolderPath, false);
                                //删除上传的文件
                                System.IO.File.Delete(filePath);
                                relaFileNames.AddRange(GetRelaFileName(compressFolderPath, commonConfigInfo.FileStoragePath));
                            }
                            else
                            {
                                relaFileNames.Add(Common.Utils.StringUtil.Combine(relaFolderPath, Utils.StringUtil.GetFileName(file.FileName)));
                            }
                        }
                        if (isEncrypt)
                        {
                            relaFileNames = relaFileNames.Select(p => CommonBL.ConvertResourceURL(p, URLOperations.Browse)).ToList();
                        }
                        return relaFileNames;
                    }
                    else
                    {
                        result = "未找到要上传的文件！";
                    }
                }
                catch (Exception ex)
                {
                    result = ex.Message;
                }
                return new List<string>() { result };

            }), "E00000108");
        }

        private List<string> GetRelaFileName(string folderPath, string fileStoragePath)
        {
            List<string> filePaths = new List<string>();
            if (Directory.Exists(folderPath))
            {
                // 获取当前目录中的所有文件
                string[] files = Directory.GetFiles(folderPath);
                foreach (string file in files)
                {
                    filePaths.Add(file.Replace(fileStoragePath, ""));
                }
                // 获取当前目录中的所有子目录
                string[] directories = Directory.GetDirectories(folderPath);
                // 递归读取每个子目录中的文件
                foreach (string dir in directories)
                {
                    filePaths.AddRange(GetRelaFileName(dir, fileStoragePath));
                }
            }
            return filePaths;
        }

        #endregion

        #region 胶片打印

        /// <summary>
        /// 打印胶片 - 合并并优化的方法
        /// </summary>
        /// <param name="print">打印信息</param>
        /// <returns>JsonResult</returns>
        public async Task<JsonResult> Print(PrintInfo print)
        {
            try
            {
                ServiceCommonConfigInfo folderConfig = CommonBL.GetServiceCommonConfigInfo();
                // 构建文件路径
                string filePath = ElincloudServer.Common.Utils.StringUtil.Combine(folderConfig.FileStoragePath, print.FilmBmpUrl);

                // 使用 Bitmap 并确保正确释放
                using var stream = System.IO.File.OpenRead(filePath);
                using var skCodec = SKCodec.Create(stream);
                using var bitmap = SKBitmap.Decode(skCodec);

                if (bitmap == null)
                {
                    return Json(new ClientResult
                    {
                        Success = false,
                        Message = "无法加载图像"
                    });
                }

                // 创建打印任务
                var printJob = new PrintJob("JPDY-GZZ DICOM PRINT JOB")
                {
                    CallingAE = "localAE",
                    CalledAE = print.Printer.AE,
                    RemotePort = print.Printer.Port,
                    RemoteAddress = print.Printer.IP,
                    ComposeName = print.Printer.ComposeName,
                    FilmDirection = print.FilmDirection,
                    FilmSize = print.FilmSize
                };

                // 设置会话参数
                printJob.FilmSession.IsColor = print.Printer.IsColor;
                printJob.FilmSession.MediumType = print.Printer.MediaTypeName;
                printJob.FilmSession.PrintPriority = print.Printer.PriorityName;
                printJob.FilmSession.FilmDestination = print.Printer.FilmEndName;

                // 获取 filmBox 实例并设置属性
                FilmBox filmBox = printJob.CreateFilmBox();
                filmBox.Trim = print.Printer.IsTrimName;
                filmBox.BorderDensity = print.Printer.BorderColorName;
                filmBox.EmptyImageDensity = print.Printer.ImgBackColorName;
                filmBox.SmoothingType = print.Printer.SmoothTypeName;
                filmBox.MagnificationType = print.Printer.InterpolationName;

                // 打印份数
                printJob.FilmSession.NumberOfCopies = print.PrintNum;

                // 添加图像并执行打印
                printJob.AddImage(bitmap, 0);
                printJob.ClearFilmBox();

                // 异步打印
                await printJob.Print();

                return Json(new ClientResult
                {
                    Success = true,
                    Message = "打印完成"
                });
            }
            catch (Exception ex)
            {
                // 记录错误日志
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("打印胶片时发生异常！", ex);

                return Json(new ClientResult
                {
                    Success = false,
                    Message = "E00000449"
                });
            }
        }

        /// <summary>
        /// 添加胶片打印记录信息
        /// </summary>
        /// <param name="hospitalID">医院编号</param>
        /// <param name="records">打印记录集合</param>
        /// <returns>JsonResult</returns>
        public JsonResult AddFilmPrintRecordInfo(string hospitalGroupID, FilmPrintRecordInfo info)
        {
            return Excute(new Func<object>(() =>
            {
                return BLServices.BLService.AISliceBLService.AddFilmPrintRecordInfo(hospitalGroupID, info);
            }), "E00000109");
        }

        #endregion
    }

    /// <summary>
    /// 打印任务
    /// </summary>
    public class PrintJob
    {
        #region 打印属性
        /// <summary>
        /// 本地AE
        /// </summary>
        public string CallingAE { get; set; }
        /// <summary>
        /// 服务AE
        /// </summary>
        public string CalledAE { get; set; }
        /// <summary>
        /// 服务端口
        /// </summary>
        public int RemotePort { get; set; }
        /// <summary>
        /// 服务IP
        /// </summary>
        public string RemoteAddress { get; set; }
        /// <summary>
        /// 陈列版式
        /// </summary>
        public string ComposeName { get; set; }
        /// <summary>
        /// 胶片定向
        /// </summary>
        public string FilmDirection { get; set; }
        /// <summary>
        /// 胶片规格
        /// </summary>
        public string FilmSize { get; set; }
        /// <summary>
        /// 胶片会话
        /// </summary>
        public FilmSession FilmSession { get; private set; }
        #endregion

        #region 构造函数
        // 定义胶片盒
        private FilmBox _currentFilmBox;
        /// <summary>
        /// 构造函数
        /// </summary>
        /// <param name="jobLabel">会话名称</param>
        public PrintJob(string jobLabel)
        {
            // 创建会话
            FilmSession = new FilmSession(DicomUID.BasicFilmSession)
            {
                FilmSessionLabel = jobLabel, //会话名称
                MediumType = "8INX10IN", //胶片规格
                NumberOfCopies = 1 //默认打印1份
            };
        }
        #endregion

        #region 胶片盒实例
        /// <summary>
        /// 创建胶片盒
        /// </summary>
        /// <returns></returns>
        public FilmBox CreateFilmBox()
        {
            // 创建一个新的 FilmBox 对象
            var filmBox = new FilmBox(FilmSession, null, DicomTransferSyntax.ExplicitVRLittleEndian)
            {
                ImageDisplayFormat = ComposeName,
                FilmOrientation = FilmDirection,
                FilmSizeID = FilmSize,

                MagnificationType = "NONE",
                BorderDensity = "BLACK",
                EmptyImageDensity = "BLACK"
            };

            // 初始化 FilmBox 对象
            filmBox.Initialize();

            // 将 FilmBox 添加到 BasicFilmBoxes 集合中
            FilmSession.BasicFilmBoxes.Add(filmBox);

            // 设置当前 FilmBox 为 _currentFilmBox
            _currentFilmBox = filmBox;

            // 返回创建的 FilmBox 对象
            return filmBox;
        }

        /// <summary>
        /// 清空胶片盒
        /// </summary>
        public void ClearFilmBox()
        {
            _currentFilmBox = null;
        }
        #endregion

        #region 组合位图像处理
        /// <summary>
        /// 添加图像
        /// </summary>
        /// <param name="bitmap">位图</param>
        /// <param name="index">图像索引</param>
        public void AddImage(SKBitmap bitmap, int index)
        {
            // 如果 FilmSession 是彩色的
            if (FilmSession.IsColor)
            {
                // 添加彩色图像
                AddColorImage(bitmap, index);
            }
            else
            {
                // 添加灰度图像
                AddGreyscaleImage(bitmap, index);
            }
        }

        /// <summary>
        /// 添加彩色图像
        /// </summary>
        /// <param name="bitmap">位图</param>
        /// <param name="index">索引</param>
        private void AddColorImage(SKBitmap bitmap, int index)
        {
            // 检查当前 FilmBox 是否为空
            if (_currentFilmBox == null)
            {
                throw new InvalidOperationException("Start film box first!");
            }

            // 检查图像索引是否有效
            if (index < 0 || index > _currentFilmBox.BasicImageBoxes.Count)
            {
                throw new ArgumentOutOfRangeException("Image box index out of range");
            }

            // 创建 DICOM 数据集
            var dataset = new DicomDataset();
            dataset.Add<ushort>(DicomTag.Columns, (ushort)bitmap.Width)
                .Add<ushort>(DicomTag.Rows, (ushort)bitmap.Height)
                .Add<ushort>(DicomTag.BitsAllocated, 8)
                .Add<ushort>(DicomTag.BitsStored, 8)
                .Add<ushort>(DicomTag.HighBit, 7)
                .Add(DicomTag.PixelRepresentation, (ushort)PixelRepresentation.Unsigned)
                .Add(DicomTag.PlanarConfiguration, (ushort)PlanarConfiguration.Interleaved)
                .Add<ushort>(DicomTag.SamplesPerPixel, 3)
                .Add(DicomTag.PhotometricInterpretation, PhotometricInterpretation.Rgb.Value);

            // 获取 RGB 字节数据
            var pixels = GetColorBytes(bitmap);
            var buffer = new Dicom.IO.Buffer.MemoryByteBuffer(pixels.Data);

            // 创建 DICOM 像素数据并添加帧
            var pixelData = DicomPixelData.Create(dataset, true);
            pixelData.AddFrame(buffer);

            // 设置图像序列
            var imageBox = _currentFilmBox.BasicImageBoxes[index];
            imageBox.ImageSequence = dataset;

            // 释放资源
            pixels.Dispose();
        }
        /// <summary>
        /// 将 SKBitmap 转换为 RGB 字节数组（BGR -> RGB）
        /// </summary>
        public PinnedByteArray GetColorBytes(SKBitmap bitmap)
        {
            if (bitmap.ColorType != SKColorType.Rgba8888 && bitmap.ColorType != SKColorType.Bgra8888)
            {
                throw new ArgumentException("Unsupported color type");
            }

            int width = bitmap.Width;
            int height = bitmap.Height;
            int pixelCount = width * height;

            var result = new PinnedByteArray(pixelCount * 3); // RGB: 3 bytes per pixel

            using (var pixmap = bitmap.PeekPixels())
            {
                var pixels = pixmap.GetPixelSpan<byte>();

                for (int i = 0; i < pixelCount; i++)
                {
                    int offset = i * 4;

                    byte b = pixels[offset + 0];
                    byte g = pixels[offset + 1];
                    byte r = pixels[offset + 2];
                    // byte a = pixels[offset + 3]; // skip alpha

                    result.Data[i * 3 + 0] = r;
                    result.Data[i * 3 + 1] = g;
                    result.Data[i * 3 + 2] = b;
                }
            }

            return result;
        }
        /// <summary>

        /// <summary>
        /// 添加灰色图像
        /// </summary>
        /// <param name="bitmap">位图</param>
        /// <param name="index">索引</param>
        private void AddGreyscaleImage(SKBitmap bitmap, int index)
        {
            // 检查当前 FilmBox 是否为空
            if (_currentFilmBox == null)
            {
                throw new InvalidOperationException("Start film box first!");
            }

            // 检查图像索引是否有效
            if (index < 0 || index > _currentFilmBox.BasicImageBoxes.Count)
            {
                throw new ArgumentOutOfRangeException("Image box index out of range");
            }

            // 创建 DICOM 数据集
            var dataset = new DicomDataset();
            dataset.Add<ushort>(DicomTag.Columns, (ushort)bitmap.Width)
                .Add<ushort>(DicomTag.Rows, (ushort)bitmap.Height)
                .Add<ushort>(DicomTag.BitsAllocated, 8)
                .Add<ushort>(DicomTag.BitsStored, 8)
                .Add<ushort>(DicomTag.HighBit, 7)
                .Add(DicomTag.PixelRepresentation, (ushort)PixelRepresentation.Unsigned)
                .Add(DicomTag.PlanarConfiguration, (ushort)PlanarConfiguration.Interleaved)
                .Add<ushort>(DicomTag.SamplesPerPixel, 1)
                .Add(DicomTag.PhotometricInterpretation, PhotometricInterpretation.Monochrome2.Value);

            // 获取灰度字节数据
            var pixels = GetGrayBytes(bitmap);
            var buffer = new Dicom.IO.Buffer.MemoryByteBuffer(pixels.Data);

            // 创建 DICOM 像素数据并添加帧
            var pixelData = DicomPixelData.Create(dataset, true);
            pixelData.AddFrame(buffer);

            // 设置图像序列
            var imageBox = _currentFilmBox.BasicImageBoxes[index];
            imageBox.ImageSequence = dataset;

            // 释放资源
            pixels.Dispose();
        }
        /// 将 SKBitmap 转换为灰度字节数组
        /// </summary>
        public PinnedByteArray GetGrayBytes(SKBitmap bitmap)
        {
            int width = bitmap.Width;
            int height = bitmap.Height;
            int pixelCount = width * height;

            var result = new PinnedByteArray(pixelCount); // 1 byte per pixel

            using (var pixmap = bitmap.PeekPixels())
            {
                // 获取像素数据为 Span<byte>，每个像素为 RGBA 格式（4字节）
                var pixels = pixmap.GetPixelSpan<byte>();

                for (int i = 0; i < pixelCount; i++)
                {
                    int offset = i * 4;
                    byte r = pixels[offset + 0];
                    byte g = pixels[offset + 1];
                    byte b = pixels[offset + 2];
                    //byte a = pixels[offset + 3]; // alpha，可忽略

                    // 灰度公式：r * 0.3 + g * 0.59 + b * 0.11
                    byte gray = (byte)(r * 0.3 + g * 0.59 + b * 0.11);

                    result.Data[i] = gray;
                }
            }

            return result;
        }

        ///// <summary>
        ///// 添加图像
        ///// </summary>
        ///// <param name="bitmap">位图</param>
        ///// <param name="index">图像索引</param>
        //public void AddImage(Bitmap bitmap, int index)
        //{
        //    // 如果 FilmSession 是彩色的
        //    if (FilmSession.IsColor)
        //    {
        //        // 添加彩色图像
        //        AddColorImage(bitmap, index);
        //    }
        //    else
        //    {
        //        // 添加灰度图像
        //        AddGreyscaleImage(bitmap, index);
        //    }
        //}
        ///// <summary>
        ///// 添加彩色图像
        ///// </summary>
        ///// <param name="bitmap">位图</param>
        ///// <param name="index">索引</param>
        //private void AddColorImage(Bitmap bitmap, int index)
        //{
        //    // 检查当前 FilmBox 是否为空
        //    if (_currentFilmBox == null)
        //    {
        //        throw new InvalidOperationException("Start film box first!");
        //    }

        //    // 检查图像索引是否有效
        //    if (index < 0 || index > _currentFilmBox.BasicImageBoxes.Count)
        //    {
        //        throw new ArgumentOutOfRangeException("Image box index out of range");
        //    }

        //    // 检查位图格式是否支持
        //    if (bitmap.PixelFormat != PixelFormat.Format24bppRgb && bitmap.PixelFormat != PixelFormat.Format32bppArgb
        //        && bitmap.PixelFormat != PixelFormat.Format32bppRgb)
        //    {
        //        throw new ArgumentException("Not supported bitmap format");
        //    }

        //    // 创建 DICOM 数据集
        //    var dataset = new DicomDataset();
        //    dataset.Add<ushort>(DicomTag.Columns, (ushort)bitmap.Width)
        //        .Add<ushort>(DicomTag.Rows, (ushort)bitmap.Height)
        //        .Add<ushort>(DicomTag.BitsAllocated, 8)
        //        .Add<ushort>(DicomTag.BitsStored, 8)
        //        .Add<ushort>(DicomTag.HighBit, 7)
        //        .Add(DicomTag.PixelRepresentation, (ushort)PixelRepresentation.Unsigned)
        //        .Add(DicomTag.PlanarConfiguration, (ushort)PlanarConfiguration.Interleaved)
        //        .Add<ushort>(DicomTag.SamplesPerPixel, 3)
        //        .Add(DicomTag.PhotometricInterpretation, PhotometricInterpretation.Rgb.Value);

        //    // 创建 DICOM 像素数据
        //    var pixelData = DicomPixelData.Create(dataset, true);

        //    // 获取彩色字节数组
        //    var pixels = GetColorbytes(bitmap);
        //    var buffer = new Dicom.IO.Buffer.MemoryByteBuffer(pixels.Data);

        //    // 添加帧数据到像素数据中
        //    pixelData.AddFrame(buffer);

        //    // 获取当前图像盒子并设置图像序列为新创建的数据集
        //    var imageBox = _currentFilmBox.BasicImageBoxes[index];
        //    imageBox.ImageSequence = dataset;

        //    // 释放像素数据
        //    pixels.Dispose();
        //}
        ///// <summary>
        ///// 添加灰色图像
        ///// </summary>
        ///// <param name="bitmap">位图</param>
        ///// <param name="index">索引</param>
        //private void AddGreyscaleImage(Bitmap bitmap, int index)
        //{
        //    // 检查当前 FilmBox 是否为空
        //    if (_currentFilmBox == null)
        //    {
        //        throw new InvalidOperationException("Start film box first!");
        //    }

        //    // 检查图像索引是否有效
        //    if (index < 0 || index > _currentFilmBox.BasicImageBoxes.Count)
        //    {
        //        throw new ArgumentOutOfRangeException("Image box index out of range");
        //    }

        //    // 检查位图格式是否支持
        //    if (bitmap.PixelFormat != PixelFormat.Format24bppRgb && bitmap.PixelFormat != PixelFormat.Format32bppArgb
        //        && bitmap.PixelFormat != PixelFormat.Format32bppRgb)
        //    {
        //        throw new ArgumentException("Not supported bitmap format");
        //    }

        //    // 创建 DICOM 数据集
        //    var dataset = new DicomDataset();
        //    dataset.Add<ushort>(DicomTag.Columns, (ushort)bitmap.Width)
        //        .Add<ushort>(DicomTag.Rows, (ushort)bitmap.Height)
        //        .Add<ushort>(DicomTag.BitsAllocated, 8)
        //        .Add<ushort>(DicomTag.BitsStored, 8)
        //        .Add<ushort>(DicomTag.HighBit, 7)
        //        .Add(DicomTag.PixelRepresentation, (ushort)PixelRepresentation.Unsigned)
        //        .Add(DicomTag.PlanarConfiguration, (ushort)PlanarConfiguration.Interleaved)
        //        .Add<ushort>(DicomTag.SamplesPerPixel, 1)
        //        .Add(DicomTag.PhotometricInterpretation, PhotometricInterpretation.Monochrome2.Value);

        //    // 创建 DICOM 像素数据
        //    var pixelData = DicomPixelData.Create(dataset, true);

        //    // 获取灰度字节数组
        //    var pixels = GetGreyBytes(bitmap);
        //    var buffer = new Dicom.IO.Buffer.MemoryByteBuffer(pixels.Data);

        //    // 添加帧数据到像素数据中
        //    pixelData.AddFrame(buffer);

        //    // 获取当前图像盒子并设置图像序列为新创建的数据集
        //    var imageBox = _currentFilmBox.BasicImageBoxes[index];
        //    imageBox.ImageSequence = dataset;

        //    // 释放像素数据
        //    pixels.Dispose();
        //}

        ///// <summary>
        ///// 将位图转换为灰色图像
        ///// </summary>
        ///// <param name="bitmap"></param>
        ///// <returns>PinnedByteArray</returns>
        //public PinnedByteArray GetGreyBytes(Bitmap bitmap)
        //{
        //    int bytesPerPixel;
        //    PixelFormat format = bitmap.PixelFormat;

        //    // 根据像素格式确定每个像素占用的字节数
        //    if (format == PixelFormat.Format24bppRgb) bytesPerPixel = 3;
        //    else if (format == PixelFormat.Format32bppArgb || format == PixelFormat.Format32bppRgb) bytesPerPixel = 4;
        //    else throw new ArgumentException("Unsupported pixel format");

        //    // 锁定位图数据
        //    BitmapData bitmapData = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), ImageLockMode.ReadOnly, format);
        //    int stride = bitmapData.Stride; // 每行的字节数，可能大于宽度*bytesPerPixel

        //    // 创建一个数组存储所有像素数据
        //    byte[] pixelBuffer = new byte[stride * bitmap.Height];
        //    System.Runtime.InteropServices.Marshal.Copy(bitmapData.Scan0, pixelBuffer, 0, pixelBuffer.Length);

        //    // 创建一个数组用于存放灰度值
        //    byte[] greyPixelBuffer = new byte[bitmap.Width * bitmap.Height];

        //    for (int y = 0; y < bitmap.Height; y++)
        //    {
        //        for (int x = 0; x < bitmap.Width; x++)
        //        {
        //            // 计算在buffer中的位置
        //            int position = y * stride + x * bytesPerPixel;

        //            // 获取RGB分量值
        //            int b = pixelBuffer[position];
        //            int g = pixelBuffer[position + 1];
        //            int r = pixelBuffer[position + 2];

        //            // 转换为灰度值
        //            greyPixelBuffer[y * bitmap.Width + x] = (byte)(r * 0.3 + g * 0.59 + b * 0.11);
        //        }
        //    }

        //    bitmap.UnlockBits(bitmapData);

        //    // 假设 PinnedByteArray 类有一个接受字节数组的构造函数
        //    return new PinnedByteArray(greyPixelBuffer);
        //}

        ///// <summary>
        ///// 将位图转换为彩色图像
        ///// </summary>
        ///// <param name="bitmap">位图</param>
        ///// <returns>PinnedByteArray</returns>
        //public PinnedByteArray GetColorbytes(Bitmap bitmap)
        //{
        //    int width = bitmap.Width;
        //    int height = bitmap.Height;
        //    int pixelCount = width * height;

        //    // 创建用于存储 RGB 数据的数组（每像素 3 字节）
        //    var pixels = new PinnedByteArray(pixelCount * 3);

        //    // 锁定位图
        //    var data = bitmap.LockBits(
        //        new Rectangle(0, 0, width, height),
        //        ImageLockMode.ReadOnly,
        //        bitmap.PixelFormat);

        //    int bytesPerPixel = Image.GetPixelFormatSize(bitmap.PixelFormat) / 8;
        //    int stride = data.Stride;

        //    byte[] row = new byte[stride];

        //    for (int y = 0; y < height; y++)
        //    {
        //        // 将当前行复制到托管数组中
        //        Marshal.Copy(data.Scan0 + y * stride, row, 0, stride);

        //        for (int x = 0; x < width; x++)
        //        {
        //            int srcOffset = x * bytesPerPixel;
        //            int dstOffset = (y * width + x) * 3;

        //            // 假设源格式是 BGR 或 BGRA，转换为 RGB
        //            byte b = row[srcOffset];
        //            byte g = row[srcOffset + 1];
        //            byte r = row[srcOffset + 2];

        //            pixels[dstOffset] = r; // R
        //            pixels[dstOffset + 1] = g; // G
        //            pixels[dstOffset + 2] = b; // B
        //        }
        //    }

        //    bitmap.UnlockBits(data);

        //    return pixels;
        //}
        #endregion

        #region 执行胶片打印
        /// <summary>
        /// 打印胶片
        /// 发送至胶片打印机
        /// </summary>
        /// <returns>Task</returns>
        public async Task Print()
        {
            try
            {
                // 创建 DICOM 客户端
                var dicomClient = new DicomClient(RemoteAddress, RemotePort, false, CallingAE, CalledAE);

                // 添加 FilmSession 的 NCreate 请求
                await dicomClient.AddRequestAsync(
                    new DicomNCreateRequest(FilmSession.SOPClassUID, FilmSession.SOPInstanceUID)
                    {
                        Dataset = FilmSession
                    });

                // 遍历每个 FilmBox
                foreach (var filmbox in FilmSession.BasicFilmBoxes)
                {
                    var imageBoxRequests = new List<DicomNSetRequest>();

                    // 创建 FilmBox 的 NCreate 请求
                    var filmBoxRequest = new DicomNCreateRequest(FilmBox.SOPClassUID, filmbox.SOPInstanceUID)
                    {
                        Dataset = filmbox
                    };

                    // 处理响应收到时的逻辑
                    filmBoxRequest.OnResponseReceived = (request, response) =>
                    {
                        if (response.HasDataset)
                        {
                            var seq = response.Dataset.GetSequence(DicomTag.ReferencedImageBoxSequence);
                            for (int i = 0; i < seq.Items.Count; i++)
                            {
                                var req = imageBoxRequests[i];
                                var imageBox = req.Dataset;
                                var sopInstanceUid = seq.Items[i].GetSingleValue<string>(DicomTag.ReferencedSOPInstanceUID);
                                imageBox.AddOrUpdate(DicomTag.SOPInstanceUID, sopInstanceUid);
                                req.Command.AddOrUpdate(DicomTag.RequestedSOPInstanceUID, sopInstanceUid);
                            }
                        }
                    };

                    // 添加 FilmBox 请求到 DICOM 客户端
                    await dicomClient.AddRequestAsync(filmBoxRequest);

                    // 处理每个 ImageBox
                    foreach (var image in filmbox.BasicImageBoxes)
                    {
                        var req = new DicomNSetRequest(image.SOPClassUID, image.SOPInstanceUID) { Dataset = image };

                        imageBoxRequests.Add(req);
                        await dicomClient.AddRequestAsync(req);
                    }
                }

                // 添加 FilmSession 的 NAction 请求
                await dicomClient.AddRequestAsync(new DicomNActionRequest(FilmSession.SOPClassUID, FilmSession.SOPInstanceUID, 0x0001));

                // 发送 DICOM 请求
                await dicomClient.SendAsync();
            }
            catch (Exception ex)
            {
                // 捕获异常并记录日志
                ElincloudServer.Common.Utils.LogUtil.RecordErrorLog("发送胶片至打印机失败！", ex);
            }
        }
        #endregion
    }
}