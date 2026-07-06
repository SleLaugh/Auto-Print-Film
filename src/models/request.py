#!/src/models/request.py
# # -*- coding: utf-8 -*-

"""
输入请求模型
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
from .enums import BodyPartEnum

@dataclass
class Request:
    """处理请求实体"""
    dicom_dir: str
    output_dir: str
    body_part: BodyPartEnum = BodyPartEnum.腰椎
    
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
            'body_part': self.body_part.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Request':
        return cls(
            dicom_dir=data['dicom_dir'],
            output_dir=data['output_dir'],
            body_part=BodyPartEnum(data.get('body_part', 'lumbar'))
        )
    
    def validate(self) -> tuple[bool, Optional[str]]:
        if not self.dicom_dir:
            return False, "dicom_dir 不能为空"
        if not self.output_dir:
            return False, "output_dir 不能为空"
        return True, None