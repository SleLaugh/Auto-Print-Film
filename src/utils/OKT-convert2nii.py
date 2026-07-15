"""将目录中的 DICOM、NIfTI 或 NRRD 医学图像批量转换为 NIfTI。"""

from __future__ import annotations

import argparse
import os
import traceback
import warnings

import nibabel as nib
import nrrd
import numpy as np
import SimpleITK as sitk

from okt_utils import DEBUG, fix_spacing, logger, setup_file_logger, sin_sout_prompt

setup_file_logger(__file__)
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")


def readdcm(filepath: str):
    """按 GDCM 序列顺序读取目录中的 DICOM 文件并返回三维图像。"""

    reader = sitk.ImageSeriesReader()
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    series_file_names = reader.GetGDCMSeriesFileNames(filepath)
    reader.SetFileNames(series_file_names)
    return reader.Execute()


def convert_3d(image):
    """压缩多余通道或维度，将 SimpleITK 图像整理为三维体数据。"""

    image_out = image
    image_array = sitk.GetArrayFromImage(image)
    if len(image_array.shape) > 3:
        image_array = image_array.squeeze()
        image_array = image_array[(Ellipsis, 0)]
        width, height, *_ = image_array.shape
        image_array = np.reshape(image_array, (-1, width, height))
        image_out = sitk.GetImageFromArray(image_array)
    return image_out


def _normalize_nifti_array(image_array: np.ndarray) -> np.ndarray:
    """将 NIfTI 像素数组统一整理为三维形状，同时保留原始轴顺序。"""

    if len(image_array.shape) > 3:
        image_array = image_array.squeeze()[(Ellipsis, 0)]
        width, height, *_ = image_array.shape
        image_array = np.reshape(image_array, (width, height, -1))
    elif len(image_array.shape) == 2:
        width, height = image_array.shape
        image_array = np.reshape(image_array, (width, height, 1))
    return image_array


def _save_sitk_as_nii(image, output_path: str) -> None:
    """创建输出目录并把 SimpleITK 图像写为指定 NIfTI 文件。"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sitk.WriteImage(image, output_path)


def _save_nrrd_fallback(file_path: str, output_path: str) -> bool:
    """在常规读取失败时尝试把 NRRD 转为 NIfTI，并返回是否转换成功。"""

    try:
        nrrd_array, info = nrrd.read(file_path)
        if len(nrrd_array.shape) == 4:
            nrrd_array = nrrd_array[0]
        image = sitk.GetImageFromArray(np.transpose(nrrd_array, (2, 1, 0)))
        space_directions = info.get("space directions")
        if space_directions is not None and len(space_directions) >= 3:
            spacing = []
            for direction in space_directions[-3:]:
                arr = np.asarray(direction, dtype=float)
                spacing.append(float(np.max(np.abs(arr))) if arr.size else 1.0)
            image.SetSpacing(tuple(spacing[:3]))
        _save_sitk_as_nii(image, output_path)
        return True
    except Exception:
        if DEBUG:
            traceback.print_exc()
        return False


def main(input_dir: str, output_dir: str, convert_single_file: bool = False, **kwargs) -> None:
    """
    尝试把输入目录下的医学影像尽量统一转换成 nii.gz，并尽可能保留目录结构。
    """
    del kwargs
    os.makedirs(output_dir, exist_ok=True)
    if not (os.path.exists(input_dir) and os.path.isdir(input_dir)):
        print(f"输入必须是一个目录，{input_dir}")
        return

    for root, directories, filenames in os.walk(input_dir):
        del directories
        logger.info(f"正在解析{root}...")
        sub_dir = root[root.find(input_dir) + len(input_dir) :].strip("/").strip("\\")
        dcm_save_dir = os.path.join(output_dir, os.path.dirname(sub_dir))
        file_save_dir = os.path.join(output_dir, sub_dir)

        try:
            dcm_images = convert_3d(readdcm(root))
            os.makedirs(dcm_save_dir, exist_ok=True)
            output_path = os.path.join(dcm_save_dir, f"{os.path.basename(root)}.nii.gz")
            sitk.WriteImage(dcm_images, output_path)
            logger.info(f"\tDetect as `DCM`,{root}解析成功，文件保存到{dcm_save_dir}")
        except Exception:
            if DEBUG:
                logger.info(f"\t{root}解析失败。")

        for filename in filenames:
            input_path = os.path.join(root, filename)
            try:
                if filename.endswith(".nii.gz"):
                    fname = filename[:-7]
                else:
                    fname, _ = os.path.splitext(filename)

                if filename.endswith(".nii") or filename.endswith(".nii.gz"):
                    image = nib.load(input_path)
                    image_array = np.array(image.dataobj)
                    image_array = _normalize_nifti_array(image_array)
                    os.makedirs(file_save_dir, exist_ok=True)
                    output_path = os.path.join(file_save_dir, f"{fname}.nii.gz")
                    nib.save(nib.Nifti1Image(image_array, affine=image.affine), output_path)
                    logger.info(f"\tDetect as `nii` or `nii.gz` using nibabel, {input_path}解析成功，文件保存到{file_save_dir}")
                    continue

                ext = os.path.splitext(filename)[-1].lower()
                if not convert_single_file and ext in ("", ".dcm", ".ima"):
                    continue

                image = sitk.ReadImage(input_path)
                image_array = sitk.GetArrayFromImage(image)
                if len(image_array.shape) > 3:
                    image_array = image_array[(Ellipsis, 0)]
                    width, height, *_ = image_array.shape
                    image_array = np.reshape(image_array, (width, height, -1))
                    image = fix_spacing(image, sitk.GetImageFromArray(image_array))
                os.makedirs(file_save_dir, exist_ok=True)
                output_path = os.path.join(file_save_dir, f"{fname}.nii.gz")
                sitk.WriteImage(image, output_path)
                logger.info(f"\tDetect as SimpleITK, {input_path}解析成功，文件保存到{file_save_dir}")
            except Exception as exc:
                traceback.print_exc()
                print(f"\t{input_path}解析失败。 因为：{exc}")
                output_path = os.path.join(file_save_dir, f"{fname}.nii.gz")
                os.makedirs(file_save_dir, exist_ok=True)
                if _save_nrrd_fallback(input_path, output_path):
                    logger.info(f"\tDetect as `.nrrd`,{input_path} 解析成功，文件保存到{file_save_dir}")

    logger.info("转化完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("将所有的数据转化成nii.gz格式")
    parser.add_argument("--convert_single_file", default=False, action="store_true", help="是否单独保存每一个dcm数据")
    args = parser.parse_args()
    input_dir, output_dir = sin_sout_prompt(
        "我们将尝试打包指定目录下的所有子目录到nii.gz，并且将子目录名作为打包文件名，并保持目录结构。\n"
        "程序会尽力进行数据转化，部分解析失败可忽略，待完成后检查输出目录即可。"
    )
    main(input_dir, output_dir, convert_single_file=args.convert_single_file)
    input("按任意键退出...")
