#!/src/models/request.py
# # -*- coding: utf-8 -*-

"""
输入请求模型
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
from .enums import BodyPartEnum, CheckTypeEnum, ScanThicknessEnum

@dataclass
class Request:
    """处理请求实体"""
    dicom_dir: str
    output_dir: str
    body_part: BodyPartEnum = BodyPartEnum.腰椎
    check_type: CheckTypeEnum = CheckTypeEnum.CT
    scan_thickness: ScanThicknessEnum = ScanThicknessEnum.默认
    
    # ===== 后续扩展字段（取消注释即可使用）=====
    # patient_name: str = ""
    # patient_id: str = ""
    # axial_thickness: float = 1.0
    # sagittal_thickness: float = 1.0
    # coronal_thickness: float = 1.0
    # enable_correction: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'dicom_dir': self.dicom_dir,
            'output_dir': self.output_dir,
            'body_part': self.body_part.value,
            'check_type': self.check_type.value,
            'scan_thickness': self.scan_thickness.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Request':
        return cls(
            dicom_dir=data['dicom_dir'],
            output_dir=data['output_dir'],
            body_part=BodyPartEnum(data.get('body_part', BodyPartEnum.腰椎.value)),
            check_type=CheckTypeEnum(data.get('check_type', CheckTypeEnum.CT.value)),
            scan_thickness=ScanThicknessEnum(data.get('scan_thickness', ScanThicknessEnum.默认.value))
        )
    
    def validate(self) -> tuple[bool, Optional[str]]:
        if not self.dicom_dir:
            return False, "dicom_dir 不能为空"
        if not self.output_dir:
            return False, "output_dir 不能为空"
        return True, None
