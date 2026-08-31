#!/usr/bin/env python3
"""比较两个结果目录的交付报告。"""

import argparse
import json

from src.core.result_compare import compare_result_reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", help="基准结果目录")
    parser.add_argument("right", help="待比较结果目录")
    args = parser.parse_args()

    differences = compare_result_reports(args.left, args.right)
    if differences:
        print(json.dumps(differences, ensure_ascii=False, indent=2))
        return 1
    print("结果报告一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
