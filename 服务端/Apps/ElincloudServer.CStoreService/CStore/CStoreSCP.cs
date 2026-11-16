using System.Text;
using Dicom;
using Dicom.Log;
using Dicom.Network;
using Dicom.Imaging.Codec;
using ElincloudServer.Entity;

namespace ElincloudServer.CStoreService
{
    public class CStoreSCP : DicomService, IDicomServiceProvider, IDicomCStoreProvider, IDicomCEchoProvider
    {

        private DicomTag hospitalNumberTag = new DicomTag(0x7211, 0x0010);
        private DicomTag machineKeyTag = new DicomTag(0x7211, 0x0011);
        private int currentProssessIndex = 1;
        private static readonly object lockObj = new object();

        private static readonly DicomTransferSyntax[] AcceptedTransferSyntaxes = new DicomTransferSyntax[]
        {
            DicomTransferSyntax.ExplicitVRLittleEndian,
            DicomTransferSyntax.ExplicitVRBigEndian,
            DicomTransferSyntax.ImplicitVRLittleEndian
        };

        private static readonly DicomTransferSyntax[] AcceptedImageTransferSyntaxes = new DicomTransferSyntax[]
        {
            // Lossless
            DicomTransferSyntax.JPEGLSLossless,
            DicomTransferSyntax.JPEG2000Lossless,
            DicomTransferSyntax.JPEGProcess14SV1,
            DicomTransferSyntax.JPEGProcess14,
            DicomTransferSyntax.RLELossless,
            // Lossy
            DicomTransferSyntax.JPEGLSNearLossless,
            DicomTransferSyntax.JPEG2000Lossy,
            DicomTransferSyntax.JPEGProcess1,
            DicomTransferSyntax.JPEGProcess2_4,
            // Uncompressed
            DicomTransferSyntax.ExplicitVRLittleEndian,
            DicomTransferSyntax.ExplicitVRBigEndian,
            DicomTransferSyntax.ImplicitVRLittleEndian
        };

        public CStoreSCP(INetworkStream stream, Encoding fallbackEncoding, Logger log)
            : base(stream, fallbackEncoding, log)
        {
        }

        public Task OnReceiveAssociationRequestAsync(DicomAssociation association)
        {
            if (CStoreConfigUtil.Instance.ServerAE != association.CalledAE)
            {
                return SendAssociationRejectAsync(DicomRejectResult.Permanent, DicomRejectSource.ServiceUser, DicomRejectReason.CalledAENotRecognized);
            }

            if (association.RemoteHost != CStoreConfigUtil.Instance.IP)//这里增加这个判断是为了排除本机做连接测试的时候不验证授权
            {
                string key = CStoreConfigUtil.Instance.GetDicomMachineKey(association.RemoteHost, association.CallingAE);
                if (!CStoreConfigUtil.Instance.DicomMachineDic.ContainsKey(key))
                {
                    return SendAssociationRejectAsync(DicomRejectResult.Permanent, DicomRejectSource.ServiceUser, DicomRejectReason.CallingAENotRecognized);
                }
            }

            foreach (var pc in association.PresentationContexts)
            {
                if (pc.AbstractSyntax == DicomUID.Verification)
                    pc.AcceptTransferSyntaxes(AcceptedTransferSyntaxes);
                else if (pc.AbstractSyntax.StorageCategory != DicomStorageCategory.None)
                    pc.AcceptTransferSyntaxes(AcceptedImageTransferSyntaxes);
            }
            return SendAssociationAcceptAsync(association);
        }

        public Task OnReceiveAssociationReleaseRequestAsync()
        {
            return SendAssociationReleaseResponseAsync();
        }

        public void OnReceiveAbort(DicomAbortSource source, DicomAbortReason reason)
        {
        }

        public void OnConnectionClosed(Exception exception)
        {
        }

        public DicomCStoreResponse OnCStoreRequest(DicomCStoreRequest request)
        {
            PushActionToThreadPool(new System.Threading.WaitCallback((object obj) =>
            {
                string targetFileName = request.SOPInstanceUID.UID + ".dcm";
                try
                {
                    #region 1.获取 CStore 配置信息

                    string calledAE = Association.CalledAE;
                    DicomFile file = request.File;
                    #endregion

                    #region 2.写入医院编号

                    string dicomMachineKey = CStoreConfigUtil.Instance.GetDicomMachineKey(Association.RemoteHost, Association.CallingAE);
                    var dicomMachine = CStoreConfigUtil.Instance.DicomMachineDic[dicomMachineKey];
                    string hospitalCode = dicomMachine.Hospital.Code;
                    CStoreConfigInfo config = CStoreConfigUtil.Instance.CStoreConfigInfos.FirstOrDefault(p => p.Hospital.Code == hospitalCode);
                    file.Dataset.AddOrUpdate<string>(hospitalNumberTag, hospitalCode);
                    file.Dataset.AddOrUpdate<string>(machineKeyTag, dicomMachine.MachineKey);

                    #endregion

                    #region 3.压缩文件


                    //1.如果收到的图像不是压缩图像的时候才对图像进行压缩

                    if (!file.FileMetaInfo.TransferSyntax.IsEncapsulated)
                    {
                        if (config.CompressDCM)
                        {
                            file = ChangeSyntax(file, config.SyntaxType, config.SyntaxQuality);
                        }
                    }
                    //2.如果收到的图像是压缩图像,则根据配置进行解压处理
                    else
                    {
                        if (config.DeCompressDCM)
                        {
                            //Transfer Syntax“作用域”，即其能够影响的范围：DICOM协议规定包含头信息（File Meta Information）的文件，头信息（即group=0002）的所有元素默认采用Explicit VR Little Endian存储
                            file = file.Clone(DicomTransferSyntax.ExplicitVRLittleEndian);
                        }
                    }
                    #endregion

                    #region 4.接收文件到缓存目录

                    string fileFullName = ElincloudServer.Common.Utils.StringUtil.Combine(CStoreConfigUtil.Instance.RecieveFilePath, System.Guid.NewGuid().ToString()) + ".dcm";

                    file.Save(fileFullName);

                    #endregion

                    #region 5.移动文件到目标路径

                    string targetFolder = string.Empty;
                    lock (lockObj)
                    {
                        targetFolder = Common.Utils.StringUtil.Combine(CStoreConfigUtil.Instance.ImageSourcePath, currentProssessIndex.ToString());
                        if (currentProssessIndex < CStoreConfigUtil.Instance.ArchiveProssessCount)
                        {
                            currentProssessIndex = currentProssessIndex + 1;
                        }
                        else
                        {
                            currentProssessIndex = 1;
                        }
                    }

                    string targetFileFullName = ElincloudServer.Common.Utils.StringUtil.Combine(targetFolder, targetFileName);

                    ElincloudServer.Common.Utils.FileUtil.MoveFile(fileFullName, targetFileFullName);

                    #endregion

                    #region 6.图像转发


                    if (config.DicomImageTransferEnabled)
                    {
                        try
                        {
                            var storeRequest = new DicomCStoreRequest(file);
                            // !!! there is a Bug in fo-dicom 3.0.2 that the OnResponseReceived handlers are invoked not until the DicomClient has already
                            //     sent all the instances. So the counters are not increased image by image sent but only once in a bulk after all storage
                            //     has been finished. This bug will be fixed hopefully soon.
                            storeRequest.OnResponseReceived += (req, resp) =>
                            {
                                if (resp.Status == DicomStatus.Success)
                                {

                                }
                                else
                                {
                                    CStoreConfigUtil.Instance.PrintLogAction($"Dicom转发文件[{file}]失败！", true);
                                }
                            };
                            CStoreConfigUtil.Instance.DCMTransferClient.AddRequestAsync(storeRequest).Wait();
                            CStoreConfigUtil.Instance.DCMTransferClient.SendAsync();
                        }
                        catch (Exception ex)
                        {
                            CStoreConfigUtil.Instance.PrintLogAction($"Dicom转发文件[{file}]异常！", true, null, ex);
                        }
                    }

                    #endregion
                }
                catch (Exception ex)
                {
                    CStoreConfigUtil.Instance.PrintLogAction($"接收DCM图像失败!CallingAE：[{Association.CallingAE}] , SOPInstanceUID：[{request.SOPInstanceUID.UID}]", true, null, ex);
                }
            }));
            return new DicomCStoreResponse(request, DicomStatus.Success);
        }

        public void OnCStoreRequestException(string tempFileName, Exception ex)
        {
            // let library handle logging and error response
            CStoreConfigUtil.Instance.PrintLogAction($"接收图像[{tempFileName}]出现异常", true, null, ex);
        }

        public DicomCEchoResponse OnCEchoRequest(DicomCEchoRequest request)
        {
            //本机自己发送的CECHO 不用提示
            if (Association.RemoteHost != CStoreConfigUtil.Instance.IP)
            {
                CStoreConfigUtil.Instance.PrintLogAction($"收到来自 AE [{Association.CallingAE}] , IP [{Association.RemoteHost}] 的CEcho请求！");
            }
            return new DicomCEchoResponse(request, DicomStatus.Success);
        }

        private DicomFile ChangeSyntax(DicomFile file, DicomTransferSyntaxTypes syntaxType, int value)
        {
            if (syntaxType == DicomTransferSyntaxTypes.无损压缩_RLE_Lossless)
            {
                return file.Clone(DicomTransferSyntax.JPEGLSLossless);
            }
            else if (syntaxType == DicomTransferSyntaxTypes.无损压缩_RLE_Lossless)
            {
                return file.Clone(DicomTransferSyntax.RLELossless);
            }
            return null;
        }

        public void PushActionToThreadPool(WaitCallback callback)
        {
            ThreadPool.QueueUserWorkItem(callback);
            //这里的sleep为防止外部代码在一个for循环将资源全部占用的问题
            Thread.Sleep(1);
        }
    }
}
