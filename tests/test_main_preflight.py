import sys
import unittest
from unittest import mock

import main


class MainFrameworkIntegrationTest(unittest.TestCase):
    def test_parse_arguments_rejects_removed_overwrite_flag(self):
        argv = [
            "main.py",
            "-i", "/tmp/input",
            "-o", "/tmp/output",
            "--overwrite",
        ]

        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit):
                main.parse_arguments()


if __name__ == "__main__":
    unittest.main()
