"""响应对象的外部字段契约测试。"""

import unittest

from src.core.context import ProcessingContext
from src.core.response import Response
from src.models import Request


class ResponseContractTest(unittest.TestCase):
    """验证响应对象对外暴露的稳定字段。"""

    def _make_response(self) -> Response:
        context = ProcessingContext(
            Request(dicom_dir="/tmp/dicom", output_dir="/tmp/output")
        )
        context.set_result(
            "reconstruction",
            {
                "axial": {"count": 2},
                "sagittal": {"count": 3},
                "coronal": {"count": 4},
            },
        )
        context.add_error("planning", ValueError("规划失败"))
        context.warnings.append({"stage": "planning", "message": "使用默认范围"})
        return Response(
            success=False,
            request=context.request,
            context=context,
            total_time=1.25,
        )

    def test_to_dict_contains_legacy_counts_and_lifecycle_fields(self):
        response = self._make_response()

        data = response.to_dict()

        self.assertEqual(data["output_dir"], "/tmp/output")
        self.assertEqual(data["total_time"], 1.25)
        self.assertFalse(data["success"])
        self.assertEqual(data["axial_count"], 2)
        self.assertEqual(data["sagittal_count"], 3)
        self.assertEqual(data["coronal_count"], 4)
        self.assertEqual(
            data["errors"],
            [{"stage": "planning", "message": "规划失败"}],
        )
        self.assertEqual(
            data["warnings"],
            [{"stage": "planning", "message": "使用默认范围"}],
        )
        self.assertEqual(data["status"], "pending")

    def test_missing_reconstruction_defaults_all_plane_counts_to_zero(self):
        context = ProcessingContext(
            Request(dicom_dir="/tmp/dicom", output_dir="/tmp/output")
        )
        response = Response(success=True, request=context.request, context=context)

        self.assertEqual(response.get_axial_count(), 0)
        self.assertEqual(response.get_sagittal_count(), 0)
        self.assertEqual(response.get_coronal_count(), 0)


if __name__ == "__main__":
    unittest.main()
