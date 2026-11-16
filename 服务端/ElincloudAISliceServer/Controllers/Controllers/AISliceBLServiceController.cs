using ElincloudAISliceServer.DBEntity;
using ElincloudAISliceServer.Entity;
using ElincloudServer.BLServices;
using ElincloudServer.CommonBLService;
using ElincloudServer.Controllers;
using ElincloudServer.DBService;
using ElincloudServer.Entity;
using Microsoft.AspNetCore.Mvc;
using RISExaminationInfo = ElincloudAISliceServer.Entity.RISExaminationInfo;

namespace ElincloudAISliceServer.Controllers
{
    public class AISliceBLServiceController : BaseController
    {
        #region 智能切割影像配置
        public JsonResult GetAIDicomSliceConfigInfo()
        {
            return Excute(new Func<AIDicomSliceConfigInfo>(() =>
            {
                return BLService.AISliceBLService.GetAIDicomSliceConfigInfo();
            }), "获取智能切割影像配置异常");
        }

        public JsonResult SetAIDicomSliceConfigInfo(AIDicomSliceConfigInfo configInfo)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.AISliceBLService.SetAIDicomSliceConfigInfo(configInfo);
            }), "保存智能切割影像配置异常");
        }
        #endregion

        #region AI切割预设配置
        public JsonResult GetAIDicomSlicePresetConfigInfo(string hospitalCode)
        {
            return Excute(new Func<AIDicomSlicePresetConfigInfo>(() =>
            {
                return BLService.AISliceBLService.GetAIDicomSlicePresetConfigInfo(hospitalCode);
            }), "获取AI切割预设配置异常");
        }

        public JsonResult SetAIDicomSlicePresetConfigInfo(AIDicomSlicePresetConfigInfo configInfo)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.AISliceBLService.SetAIDicomSlicePresetConfigInfo(configInfo);
            }), "保存AI切割预设配置异常");
        }
        #endregion

        #region 切片方案
        public JsonResult GetSlicePlanConfigInfoByID(string hospitalGroupID, string id)
        {
            return Excute<SlicePlanConfigInfo>(new Func<PermissionInfo, SlicePlanConfigInfo>((PermissionInfo pi) =>
            {
                return BLService.AISliceBLService.GetSlicePlanConfigInfoByID(hospitalGroupID, id);
            }), "根据ID获取切片方案异常");
        }

        public JsonResult GetSlicePlanConfigInfos(string hospitalGroupID)
        {
            return Excute(new Func<List<SlicePlanConfigInfo>>(() =>
            {
                return BLService.AISliceBLService.GetSlicePlanConfigInfos(hospitalGroupID);
            }), "获取切片方案列表异常");
        }

        public JsonResult GetSlicePlanConfigInfosByPage(string hospitalGroupID, ControllerParameter<int> pageIndex, ControllerParameter<int> pageSize, string searchString, List<string> studyTypeNames, List<string> studyPartNames, List<string> slicePartNames)
        {
            return Excute<Page<SlicePlanConfigInfo>>(new Func<PermissionInfo, Page<SlicePlanConfigInfo>>((PermissionInfo pi) =>
            {
                return BLService.AISliceBLService.GetSlicePlanConfigInfosByPage(hospitalGroupID, pageIndex.Value, pageSize.Value, searchString, studyTypeNames, studyPartNames, slicePartNames);
            }), "分页获取切片方案列表异常");
        }


        public JsonResult AddSlicePlanConfigInfo(string hospitalGroupID, SlicePlanConfigInfo info)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.AISliceBLService.AddSlicePlanConfigInfo(pi, hospitalGroupID, info);
            }), "新增切片方案异常");
        }

        public JsonResult UpdateSlicePlanConfigInfo(string hospitalGroupID, SlicePlanConfigInfo info)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.AISliceBLService.UpdateSlicePlanConfigInfo(pi, hospitalGroupID, info);
            }), "修改切片方案异常");
        }

        public JsonResult DeleteSlicePlanConfigInfo(string hospitalGroupID, string ID)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.AISliceBLService.DeleteSlicePlanConfigInfo(hospitalGroupID, ID);
            }), "删除切片方案异常");
        }

        public JsonResult UpdateSlicePlanConfigStatus(string hospitalGroupID, string ID, ControllerParameter<bool> isEnabel)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.AISliceBLService.UpdateSlicePlanConfigStatus(hospitalGroupID, ID, isEnabel.Value);
            }), "更新切割方案启用状态异常");
        }

        #endregion

        #region 智能排片模版
        public JsonResult GetLayoutTemplateInfoByID(string hospitalGroupID, string id)
        {
            return Excute(new Func<LayoutTemplateInfo>(() =>
            {
                return BLService.AISliceBLService.GetLayoutTemplateInfoByID(hospitalGroupID, id);
            }), "根据ID获取智能排片模版异常");
        }

        public JsonResult GetLayoutTemplateInfos(string hospitalGroupID)
        {
            return Excute(new Func<List<LayoutTemplateInfo>>(() =>
            {
                return BLService.AISliceBLService.GetLayoutTemplateInfos(hospitalGroupID);
            }), "获取智能排片模版列表异常");
        }

        public JsonResult GetLayoutTemplateInfosByPage(string hospitalGroupID, ControllerParameter<int> pageIndex, ControllerParameter<int> pageSize, string searchString, List<string> studyTypeNames, List<string> studyPartNames)
        {
            return Excute(new Func<Page<LayoutTemplateInfo>>(() =>
            {
                return BLService.AISliceBLService.GetLayoutTemplateInfosByPage(hospitalGroupID, pageIndex.Value, pageSize.Value, searchString, studyTypeNames, studyPartNames);
            }), "分页获取智能排片模版异常");
        }


        public JsonResult AddLayoutTemplateInfo(string hospitalGroupID, LayoutTemplateInfo info)
        {
            return Excute(new Action(() =>
            {
                BLService.AISliceBLService.AddLayoutTemplateInfo(hospitalGroupID, info);
            }), "新增智能排片模版异常");
        }

        public JsonResult UpdateLayoutTemplateInfo(string hospitalGroupID, LayoutTemplateInfo info)
        {
            return Excute(new Action(() =>
            {
                BLService.AISliceBLService.UpdateLayoutTemplateInfo(hospitalGroupID, info);
            }), "修改智能排片模版异常");
        }

        public JsonResult DeleteLayoutTemplateInfo(string hospitalGroupID, string ID)
        {
            return Excute(new Action(() =>
            {
                BLService.AISliceBLService.DeleteLayoutTemplateInfo(hospitalGroupID, ID);
            }), "删除智能排片模版异常");
        }

        public JsonResult UpdateLayoutTemplateStatus(string hospitalGroupID, string ID, ControllerParameter<bool> isEnabel)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.AISliceBLService.UpdateLayoutTemplateStatus(hospitalGroupID, ID, isEnabel.Value);
            }), "更新排片模板启用状态异常");
        }
        #endregion

        #region AI切割结果推送配置
        public JsonResult GetAISliceResultPushConfigInfo(string hospitalCode)
        {
            return Excute(new Func<AISliceResultPushConfigInfo>(() =>
            {
                return BLService.AISliceBLService.GetAISliceResultPushConfigInfo(hospitalCode);
            }), "获取AI切割结果推送配置异常");
        }

        public JsonResult SetAISliceResultPushConfigInfo(AISliceResultPushConfigInfo configInfo)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.AISliceBLService.SetAISliceResultPushConfigInfo(configInfo);
            }), "保存AI切割结果推送配置异常");
        }
        #endregion

        #region 智能排片业务相关
        /// <summary>
        /// 分页获取检查列表
        /// </summary>
        /// <param name="hospitalNumber"></param>
        /// <param name="hospitalGroupID"></param>
        /// <param name="searchString"></param>
        /// <param name="searchParameterList"></param>
        /// <param name="pageIndex"></param>
        /// <param name="pageSize"></param>
        /// <param name="orderByParameterList"></param>
        /// <returns></returns>
        public JsonResult GetExaminationInfosByPage(string hospitalName, string hospitalNumber, string hospitalGroupID, ControllerParameter<DateTime> startDate, ControllerParameter<DateTime> endDate, string searchString, List<string> studyTypeNames, ControllerParameter<int> pageIndex, ControllerParameter<int> pageSize)
        {
            return Excute(new Func<Page<RISExaminationInfo>>(() =>
            {
                return BLService.AISliceBLService.GetExaminationInfosByPage(hospitalName, hospitalNumber, hospitalGroupID, startDate.Value, endDate.Value, searchString, studyTypeNames, pageIndex.Value, pageSize.Value);
            }), "分页获取检查列表异常");
        }

        /// <summary>
        /// 重新切割
        /// </summary>
        /// <param name="pi"></param>
        /// <param name="hospitalGroupID"></param>
        /// <param name="examinationID"></param>
        public JsonResult RestartSlice(string hospitalGroupID, string examinationID)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.AISliceBLService.RestartSlice(pi, hospitalGroupID, examinationID);
            }), "重新切割异常");
        }

        /// <summary>
        /// 加急处理
        /// </summary>
        /// <param name="hospitalGroupID"></param>
        /// <param name="examinationID"></param>
        /// <param name="examinationID"></param>
        public JsonResult ProcessUrgent(string hospitalGroupID, string examinationID, bool isUrgent)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.AISliceBLService.ProcessUrgent(pi, hospitalGroupID, examinationID, isUrgent);
            }), "加急处理异常");
        }

        /// <summary>
        /// 重新切割序列
        /// </summary>
        /// <param name="pi"></param>
        /// <param name="hospitalGroupID"></param>
        /// <param name="examinationID"></param>
        public JsonResult RestartSliceSerise(string hospitalGroupID, string examinationID, string seriesInstanceUID)
        {
            return Excute(new Action<PermissionInfo>((PermissionInfo pi) =>
            {
                BLService.AISliceBLService.RestartSliceSerise(pi, hospitalGroupID, examinationID, seriesInstanceUID);
            }), "重新切割序列异常");
        }

        /// <summary>
        /// 根据检查号获取检查信息
        /// </summary>
        /// <param name="hospitalGroupID"></param>
        /// <param name="examinationID"></param>
        /// <returns></returns>
        public JsonResult GetExaminationInfoByID(string hospitalGroupID, string examinationID)
        {
            return Excute(new Func<RISExaminationInfo>(() =>
            {
                return BLService.AISliceBLService.GetExaminationInfoByID(hospitalGroupID, examinationID);
            }), "根据检查号获取检查信息异常");
        }

        /// <summary>
        /// 根据检查号获取智能切片记录
        /// </summary>
        /// <param name="hospitalGroupID"></param>
        /// <param name="examinationID"></param>
        /// <returns></returns>
        public JsonResult GetAIFilmSliceRecordInfos(string hospitalGroupID, string examinationID)
        {
            return Excute(new Func<List<AIFilmSliceRecordInfo>>(() =>
            {
                return BLService.AISliceBLService.GetAIFilmSliceRecordInfos(hospitalGroupID, examinationID);
            }), "根据检查号获取智能切片记录异常");
        }

        /// <summary>
        /// 根据序列UID获取切片模板信息
        /// </summary>
        /// <param name="hospitalGroupID"></param>
        /// <param name="seriesInstanceUID"></param>
        /// <returns></returns>
        public JsonResult GetLayoutTemplateInfosBySeriesID(string hospitalGroupID, string seriesInstanceUID)
        {
            return Excute(new Func<List<LayoutTemplateInfo>>(() =>
            {
                return BLService.AISliceBLService.GetLayoutTemplateInfosBySeriesID(hospitalGroupID, seriesInstanceUID);
            }), "根据序列UID获取切片模板信息异常");
        }

        /// <summary>
        /// 根据模版ID获取切割方案结果
        /// </summary>
        /// <param name="hospitalGroupID"></param>
        /// <param name="layoutTemplateID"></param>
        /// <returns></returns>
        public JsonResult GetAIFilmSliceResultInfosByLayoutTemplateID(string hospitalGroupID, string layoutTemplateID, string examinationID, string seriesInstanceUID)
        {
            return Excute(new Func<List<AIFilmSliceResultInfo>>(() =>
            {
                return BLService.AISliceBLService.GetAIFilmSliceResultInfosByLayoutTemplateID(hospitalGroupID, layoutTemplateID, examinationID, seriesInstanceUID);
            }), "根据模版ID获取切割方案结果异常");
        }
        #endregion
    }
}
