"""按模型约定置换/翻转 NIfTI 存储轴，推理后无插值恢复源图空间。"""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import SimpleITK as sitk


@dataclass(frozen=True)
class ImageGeometry:
    size: tuple[int, ...]
    spacing: tuple[float, ...]
    origin: tuple[float, ...]
    direction: tuple[float, ...]

    @classmethod
    def from_image(cls, image: sitk.Image) -> "ImageGeometry":
        if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError("模型方向适配只支持三维标量图像")
        return cls(image.GetSize(), image.GetSpacing(), image.GetOrigin(), image.GetDirection())

    @classmethod
    def from_file(cls, path: Path) -> "ImageGeometry":
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(path))
        reader.ReadImageInformation()
        if reader.GetDimension() != 3 or reader.GetNumberOfComponents() != 1:
            raise ValueError("模型方向适配只支持三维标量图像")
        return cls(reader.GetSize(), reader.GetSpacing(), reader.GetOrigin(), reader.GetDirection())

    def apply(self, image: sitk.Image) -> None:
        image.SetSpacing(self.spacing)
        image.SetOrigin(self.origin)
        image.SetDirection(self.direction)

    def validate(self, image: sitk.Image, description: str) -> None:
        actual = self.from_image(image)
        if actual.size != self.size or any(
            not np.allclose(left, right, rtol=0, atol=1e-5)
            for left, right in (
                (actual.spacing, self.spacing), (actual.origin, self.origin),
                (actual.direction, self.direction),
            )
        ):
            raise ValueError(f"{description}空间不一致，拒绝发布分割结果")


@dataclass(frozen=True)
class OrientationTransform:
    original: ImageGeometry
    normalized: ImageGeometry
    permute_order: tuple[int, ...]
    flip_axes: tuple[bool, ...]
    orientation: str
    normalized_exact: ImageGeometry | None = None


def orient_image_for_inference(
    image: sitk.Image, orientation: str,
) -> tuple[sitk.Image, OrientationTransform]:
    """仅重排体素，不旋转斜扫角度或改变任何体素对应的物理位置。"""

    if orientation not in {"RAS", "LPS"}:
        raise ValueError("input_orientation 仅支持 RAS/LPS")
    original = ImageGeometry.from_image(image)
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation(orientation)
    normalized = orienter.Execute(image)
    transform = OrientationTransform(
        original, ImageGeometry.from_image(normalized),
        tuple(orienter.GetPermuteOrder()), tuple(orienter.GetFlipAxes()), orientation,
    )
    return normalized, transform


def restore_segmentation_image(
    segmentation: sitk.Image, transform: OrientationTransform,
) -> sitk.Image:
    """验证网络输出后逆置换/翻转；不允许用复制头信息掩盖空间错误。"""

    transform.normalized.validate(segmentation, "nnUNet 输出与方向规范化输入")
    if transform.normalized_exact is not None:
        # NIfTI-1 的头信息使用 float32；已验证网络忠实保持实际输入头之后，
        # 恢复写盘前的精确几何，再逆轴变换，避免毫米大原点的舍入累积。
        segmentation = sitk.Image(segmentation)
        transform.normalized_exact.apply(segmentation)
    restored = sitk.Flip(segmentation, list(transform.flip_axes), False)
    inverse_order = [transform.permute_order.index(axis) for axis in range(3)]
    restored = sitk.PermuteAxes(restored, inverse_order)
    transform.original.validate(restored, "逆方向适配输出与原 CT")
    # 只消除运算/NIfTI 头往返的小量浮点误差，体素位置已在上面独立验证。
    transform.original.apply(restored)
    return restored


def prepare_oriented_input(
    input_path: Path, directory: Path, orientation: str,
) -> tuple[Path, OrientationTransform]:
    """规范化 CT 仅写到本次推理工作目录，不覆盖或发布原预处理缓存。"""

    image = sitk.ReadImage(str(input_path))
    normalized, transform = orient_image_for_inference(image, orientation)
    directory.mkdir(parents=True, exist_ok=True)
    normalized_path = directory / input_path.name
    sitk.WriteImage(normalized, str(normalized_path))
    return normalized_path, replace(
        transform, normalized=ImageGeometry.from_file(normalized_path),
        normalized_exact=transform.normalized,
    )


def restore_segmentation_file(path: Path, transform: OrientationTransform) -> None:
    """工作目录中的分割完成空间校验与恢复后，才可交给缓存发布。"""

    segmentation = sitk.ReadImage(str(path))
    restored = restore_segmentation_image(segmentation, transform)
    sitk.WriteImage(restored, str(path))
