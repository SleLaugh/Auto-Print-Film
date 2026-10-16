"""Executable entry point, including a private build-time packaging smoke check."""

import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile

from src.runtime import bootstrap_native_threads

_thread_bootstrap_error = None
if __name__ == "__main__":
    try:
        bootstrap_native_threads()
    except ValueError as error:
        # The normal request entry writes report.json after configuring logs.
        _thread_bootstrap_error = error

from main import main


def check_packaging():
    """Exercise bundled native/image libraries without models or patient data."""
    import numpy as np
    import SimpleITK as sitk
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage
    from src.config import load_app_config
    from src.models.window import load_window_presets
    from src.runtime import application_root, config_path

    assert getattr(sys, "frozen", False), "Expected a frozen executable"
    assert importlib.util.find_spec("torch") is None
    assert importlib.util.find_spec("nnunetv2") is None
    array = np.arange(16, dtype=np.uint16).reshape(4, 4)
    with tempfile.TemporaryDirectory(prefix="autoprintfilm-library-check-") as directory:
        path = Path(directory) / "synthetic.nii.gz"
        sitk.WriteImage(sitk.GetImageFromArray(array), str(path))
        assert np.array_equal(sitk.GetArrayFromImage(sitk.ReadImage(str(path))), array)
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = "1.2.826.0.1.3680043.10.999.1"
    dataset = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.SpecificCharacterSet = "ISO_IR 192"
    dataset.PatientName = "打包验证"
    dataset.Rows, dataset.Columns = array.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = array.tobytes()
    stream = io.BytesIO()
    pydicom.dcmwrite(stream, dataset, enforce_file_format=True)
    stream.seek(0)
    loaded = pydicom.dcmread(stream)
    assert np.array_equal(loaded.pixel_array, array)
    assert str(loaded.PatientName) == "打包验证"
    config = load_app_config()
    presets = load_window_presets()
    print(json.dumps({
        "frozen": True,
        "application_root": str(application_root()),
        "config_dir": str(config_path("logger_config.json").parent),
        "logging_base_dir": config.logging.base_dir,
        "window_presets": {name: value.to_dict() for name, value in presets.items()},
        "inference_environment": "external",
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--_build-check"]:
        if _thread_bootstrap_error is not None:
            raise _thread_bootstrap_error
        sys.exit(check_packaging())
    sys.exit(main(startup_error=_thread_bootstrap_error))
