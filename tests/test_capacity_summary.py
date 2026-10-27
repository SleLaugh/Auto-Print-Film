"""Synthetic offline logs verify concurrency, cache modes and sampled-resource limits."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/summarize_capacity.py"
SPEC = importlib.util.spec_from_file_location("summarize_capacity", SCRIPT)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def event(run, name, at, *, boot="boot-a", **details):
    return {
        "schema_version": 1, "event": name, "run_id": run, "boot_id": boot,
        "pid": 123, "wall_time": "2026-10-27T00:00:00+00:00", "monotonic_s": at,
        "series_key": "hashed-series", "slice_part": "腰椎", "study_type": "CT",
        **details,
    }


def append_records(root, name, records, *, prefix="[Capacity] "):
    path = root / name / "log/DebugLog.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write("timestamp " + prefix + json.dumps(record, ensure_ascii=False) + "\n")
    return path


def task(root, run, start, end, *, boot="boot-a", mode="direct_inference", status="succeeded",
         part="腰椎", **end_details):
    records = []
    if start is not None:
        records.append(event(run, "task_start", start, boot=boot, slice_part=part))
    if end is not None:
        records.append(event(
            run, "task_end", end, boot=boot, slice_part=part, status=status,
            duration_seconds=end - start if start is not None else 4,
            model_id="292", segmentation_mode=mode, cache_wait_seconds=0, **end_details,
        ))
    append_records(root, run, records)
    return records


def host(at, *, boot="boot-a", cpu=0, available=0, gpu=None):
    return {
        "schema_version": 1, "timestamp_utc": "2026-10-27T00:00:00+00:00",
        "monotonic_s": at, "boot_id": boot,
        "cpu": {"utilization_percent": cpu} if cpu is not None else None,
        "memory": {"total_bytes": 100 * summary.MIB,
                   "available_bytes": available * summary.MIB} if available is not None else None,
        "gpu": gpu,
    }


def write_hosts(root, records):
    root.mkdir()
    (root / "host-2026-10-27.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8",
    )


def test_real_half_open_overlap_groups_modes_statuses_stages_and_inference_attempts(tmp_path):
    task(tmp_path, "a", 0, 10, stage_seconds={"segmentation": 4, "reconstruction": 3})
    task(tmp_path, "b", 5, 15, mode="cache_hit", part="胸椎", status="failed")
    task(tmp_path, "c", 15, 20, mode="wait_reuse", status="aborted")
    append_records(tmp_path, "a", [
        event("a", "inference_start", 1, model_id="292", attempt=1),
        event("a", "inference_end", 2, model_id="292", attempt=1, outcome="failed", duration_seconds=1),
        event("a", "inference_start", 2, model_id="292", attempt=2),
        event("a", "inference_end", 4, model_id="292", attempt=2, outcome="succeeded", duration_seconds=2),
    ])
    report = summary.build_report(tmp_path)
    assert report["tasks"]["sample_count"] == 3
    assert report["tasks"]["completed_count"] == 3
    assert report["tasks"]["succeeded_count"] == 1
    assert report["tasks"]["failed_count"] == 1
    assert report["tasks"]["aborted_count"] == 1
    assert report["tasks"]["max_observed_overlap"] == 2
    groups = {group["segmentation_mode"]: group for group in report["task_groups"]}
    assert groups["direct_inference"]["actual_max_overlap_during_task"] == 2
    assert groups["cache_hit"]["actual_max_overlap_during_task"] == 2
    assert groups["wait_reuse"]["actual_max_overlap_during_task"] == 1
    assert groups["direct_inference"]["stage_seconds"]["segmentation"]["p50"] == 4
    assert groups["cache_hit"]["stage_seconds"] == {}
    assert groups["cache_hit"]["cache_wait_seconds"]["count"] == 1
    assert groups["cache_hit"]["cache_wait_seconds"]["p95"] == 0
    assert report["inference"]["max_observed_overlap"] == 1
    assert report["inference"]["outcomes"] == {"failed": 1, "succeeded": 1}
    assert report["recorded_boot_windows"][0]["task_exposure_seconds_by_overlap"] == [
        {"actual_overlap": 1, "seconds": 15}, {"actual_overlap": 2, "seconds": 5},
    ]
    assert report["resources_by_actual_overlap"] == []


def test_missing_ends_boot_separation_and_copied_events_do_not_invent_overlap(tmp_path):
    records = task(tmp_path, "a", 0, 10)
    append_records(tmp_path, "copy", records)
    task(tmp_path, "b", 0, 10, boot="boot-b")
    task(tmp_path, "unfinished", 0, None)
    task(tmp_path, "end-only", None, 20)
    task(tmp_path, "unknown-boot", 0, 10, boot=None)
    append_records(tmp_path, "unfinished", [event(
        "unfinished", "inference_start", 1, model_id="292", attempt=1,
    )])
    report = summary.build_report(tmp_path)
    assert report["tasks"]["sample_count"] == 5
    assert report["tasks"]["completed_count"] == 4
    assert report["tasks"]["missing_start_count"] == 1
    assert report["tasks"]["missing_end_count"] == 1
    assert report["tasks"]["max_observed_overlap"] == 1
    assert report["issues"]["duplicate_capacity_events"] == 2
    assert report["inference"]["missing_end_count"] == 1
    assert report["inference"]["max_observed_overlap"] is None
    assert len(report["recorded_boot_windows"]) == 2
    assert report["recorded_boot_windows"][0]["completed_per_hour_in_recorded_window"] == 360
    assert any(group["actual_max_overlap_during_task"] is None for group in report["task_groups"])


def test_empty_data_and_measured_zero_remain_different(tmp_path):
    empty = summary.build_report(tmp_path)
    assert empty["tasks"]["max_observed_overlap"] is None
    assert empty["tasks"]["runtime_seconds"]["count"] == 0
    assert empty["tasks"]["runtime_seconds"]["p50"] is None
    task(tmp_path, "instant", 1, 1)
    report = summary.build_report(tmp_path)
    assert report["tasks"]["max_observed_overlap"] == 0
    assert report["tasks"]["runtime_seconds"]["count"] == 1
    assert report["tasks"]["runtime_seconds"]["p95"] == 0
    assert report["recorded_boot_windows"][0]["completed_per_hour_in_recorded_window"] is None


def test_host_zero_missing_cross_boot_and_gpu_cache_uses_actual_query_time(tmp_path):
    tasks, hosts = tmp_path / "tasks", tmp_path / "hosts"
    task(tasks, "a", 5, 15)
    task(tasks, "b", 10, 20)
    gpu = {"sampled_monotonic_s": 2, "age_s": 10,
           "devices": [{"index": 0, "memory_used_mib": 0, "memory_total_mib": 32000,
                        "utilization_percent": 0}]}
    records = [host(12, gpu=gpu), host(14, cpu=None, available=None, gpu={**gpu, "age_s": 12}),
               host(16, cpu=50, available=25), host(12, boot="another-machine")]
    records.append(records[0])
    write_hosts(hosts, records)
    report = summary.build_report(tasks, hosts)
    values = {entry["actual_recorded_task_overlap"]: entry
              for entry in report["resources_by_actual_overlap"]}
    assert report["inputs"]["host_samples"] == 4
    assert report["issues"]["duplicate_host_samples"] == 1
    assert values[2]["host_sample_count"] == 2
    assert values[2]["sampled_cpu_utilization_percent"]["count"] == 1
    assert values[2]["sampled_cpu_utilization_percent"]["max"] == 0
    assert values[2]["sampled_ram_available_mib"]["min"] == 0
    assert values[2]["sampled_ram_unavailable_mib"]["max"] == 100
    assert values[2]["gpu_unique_query_count"] == 0
    assert values[0]["host_sample_count"] == 0
    assert values[0]["sampled_cpu_utilization_percent"]["max"] is None
    assert values[0]["gpu_unique_query_count"] == 1
    assert values[0]["gpu_devices"][0]["sampled_memory_used_mib"]["count"] == 1
    assert values[0]["gpu_devices"][0]["sampled_memory_used_mib"]["max"] == 0
    assert values[1]["sampled_ram_available_mib"]["min"] == 25
    assert values[None]["host_sample_count"] == 1


def test_corrupt_lines_conflicting_boundaries_and_reversed_interval_are_reported(tmp_path):
    records = task(tmp_path, "conflict", 0, 10)
    path = append_records(tmp_path, "conflict", [{**records[0], "pid": 999}])
    with path.open("a") as stream:
        stream.write("[Capacity] not-json patient-secret\n")
        stream.write("[Capacity] []\n")
    task(tmp_path, "reversed", 10, 2)
    report = summary.build_report(tmp_path)
    assert report["issues"]["invalid_capacity_lines"] == 2
    assert report["issues"]["conflicting_capacity_event_keys"] == 1
    assert report["issues"]["reversed_task_intervals"] == 1
    assert report["tasks"]["max_observed_overlap"] is None
    assert "patient-secret" not in json.dumps(report)


def test_cli_only_reads_logs_and_writes_an_explicit_json_report_with_plain_summary(tmp_path):
    tasks = tmp_path / "tasks"
    task(tasks, "a", 0, 10)
    (tasks / "patient-secret.dcm").write_bytes(b"not a DICOM file")
    (tasks / "patient-secret.nii.gz").write_bytes(b"not a volume")
    output = tmp_path / "reports/capacity.json"
    result = subprocess.run(
        [sys.executable, "-S", str(SCRIPT), "--tasks-root", str(tasks), "--output", str(output)],
        capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["tasks"]["sample_count"] == 1
    assert "Recorded tasks=1" in result.stdout
    assert "safe concurrency limit is not inferred" in result.stdout
    assert "patient-secret" not in result.stdout + result.stderr + output.read_text(encoding="utf-8")
    assert result.stderr == ""


def test_missing_input_root_does_not_write_a_misleading_empty_report(tmp_path):
    output = tmp_path / "output.json"
    assert summary.main(["--tasks-root", str(tmp_path / "missing"), "--output", str(output)]) == 1
    assert not output.exists()


def test_percentiles_include_sample_count_and_use_observed_values():
    assert summary.stats([0, 10, 20])["p50"] == 10
    assert summary.stats([0, 10, 20])["p95"] == 19
    assert summary.stats([None])["count"] == 0
    assert summary.stats([0])["p95"] == 0
