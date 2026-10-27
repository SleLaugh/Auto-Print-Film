#!/usr/bin/env python3
"""Summarize recorded capacity intervals and optional host samples, offline.

Only **/log/DebugLog.txt and host-*.jsonl are read. No application/native
libraries are imported and no DICOM or image volumes are opened. Run explicitly:
  python3 scripts/summarize_capacity.py --tasks-root RESULTS --output capacity.json
Add --host-root HOST_LOGS to correlate sampled resources by boot/monotonic time.
This describes recorded Pipeline.run intervals, never a safe concurrency limit.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys


PREFIX = "[Capacity] "
EVENTS = {"task_start", "task_end", "inference_start", "inference_end"}
MODES = {"direct_inference", "cache_hit", "wait_reuse", "unknown"}
MIB = 1024 * 1024


def number(value, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        result = float(value)
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(result) or nonnegative and result < 0:
        return None
    return result


def stats(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return {"count": 0, "min": None, "p50": None, "p95": None, "max": None}

    def percentile(fraction):
        position = (len(values) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return round(values[lower] + (values[upper] - values[lower]) * (position - lower), 6)

    return {
        "count": len(values), "min": round(values[0], 6),
        "p50": percentile(0.5), "p95": percentile(0.95), "max": round(values[-1], 6),
    }


def boot_key(record):
    value = record.get("boot_id")
    return value if isinstance(value, str) and value else None


def read_events(root, issues, inputs):
    indexed = {}
    conflicts = set()
    for path in sorted(root.glob("**/log/DebugLog.txt")):
        inputs["task_log_files"] += 1
        try:
            with path.open(encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    offset = line.find(PREFIX)
                    if offset < 0:
                        continue
                    inputs["capacity_lines"] += 1
                    try:
                        record = json.loads(line[offset + len(PREFIX):])
                        if not isinstance(record, dict) or record.get("schema_version") != 1:
                            raise ValueError("unsupported event")
                        event = record.get("event")
                        run_id = record.get("run_id")
                        if event not in EVENTS or not isinstance(run_id, str) or not run_id:
                            raise ValueError("invalid event identity")
                        if number(record.get("monotonic_s"), nonnegative=True) is None:
                            raise ValueError("invalid event time")
                        model = attempt = None
                        if event.startswith("inference_"):
                            model = record.get("model_id")
                            attempt = record.get("attempt")
                            if not isinstance(model, str) or not model:
                                raise ValueError("invalid model")
                            if type(attempt) is not int or attempt < 1:
                                raise ValueError("invalid attempt")
                        key = (boot_key(record), run_id, event, model, attempt)
                        if key in indexed:
                            if indexed[key] == record:
                                issues["duplicate_capacity_events"] += 1
                            else:
                                conflicts.add(key)
                        else:
                            indexed[key] = record
                    except (ValueError, TypeError):
                        issues["invalid_capacity_lines"] += 1
        except OSError:
            issues["unreadable_task_logs"] += 1
    issues["conflicting_capacity_event_keys"] += len(conflicts)
    return [record for key, record in indexed.items() if key not in conflicts]


def pair_intervals(records, kind, issues):
    pairs = defaultdict(dict)
    for record in records:
        if not record["event"].startswith(kind + "_"):
            continue
        key = (boot_key(record), record["run_id"])
        if kind == "inference":
            key += (record["model_id"], record["attempt"])
        pairs[key][record["event"]] = record
    result = []
    for key, pair in pairs.items():
        first, last = pair.get(kind + "_start"), pair.get(kind + "_end")
        details = {**(first or {}), **(last or {})}
        start = number(first.get("monotonic_s")) if first else None
        end = number(last.get("monotonic_s")) if last else None
        valid = first is not None and last is not None and end >= start
        if first is not None and last is not None and not valid:
            issues["reversed_" + kind + "_intervals"] += 1
        duration = number(last.get("duration_seconds"), nonnegative=True) if last else None
        if valid and duration is None:
            duration = end - start
        if first is not None and last is not None and not valid:
            duration = None
        result.append({
            "boot_id": key[0], "details": details, "start": start, "end": end,
            "valid_interval": valid, "has_start": first is not None,
            "has_end": last is not None, "duration": duration, "max_overlap": None,
        })
    return result


class Timeline:
    def __init__(self, intervals):
        changes = defaultdict(int)
        for item in intervals:
            changes[item["start"]] += 1
            changes[item["end"]] -= 1
        self.times = sorted(changes)
        self.counts = []
        active = 0
        for timestamp in self.times:
            active += changes[timestamp]
            self.counts.append(active)

    def at(self, timestamp):
        index = bisect_right(self.times, timestamp) - 1
        return self.counts[index] if index >= 0 else 0

    def peak(self, start, end):
        if start == end:
            return 0
        left, right = bisect_left(self.times, start), bisect_left(self.times, end)
        return max(self.counts[left:right], default=0)

    def exposure(self):
        seconds = defaultdict(float)
        for index in range(len(self.times) - 1):
            seconds[self.counts[index]] += self.times[index + 1] - self.times[index]
        return [{"actual_overlap": count, "seconds": round(duration, 6)}
                for count, duration in sorted(seconds.items())]


def timelines(items):
    by_boot = defaultdict(list)
    for item in items:
        if item["valid_interval"] and item["boot_id"] is not None:
            by_boot[item["boot_id"]].append(item)
    result = {boot: Timeline(intervals) for boot, intervals in by_boot.items()}
    for item in items:
        if item["valid_interval"] and item["boot_id"] in result:
            item["max_overlap"] = result[item["boot_id"]].peak(item["start"], item["end"])
    return result


def interval_summary(items):
    statuses = Counter(str(item["details"].get("status") or "unknown")
                       for item in items if item["has_end"])
    peaks = [item["max_overlap"] for item in items if item["max_overlap"] is not None]
    return {
        "sample_count": len(items),
        "completed_count": sum(item["has_end"] for item in items),
        "succeeded_count": statuses["succeeded"], "failed_count": statuses["failed"],
        "aborted_count": statuses["aborted"],
        "unknown_status_count": sum(value for key, value in statuses.items()
                                    if key not in {"succeeded", "failed", "aborted"}),
        "missing_start_count": sum(not item["has_start"] for item in items),
        "missing_end_count": sum(not item["has_end"] for item in items),
        "max_observed_overlap": max(peaks) if peaks else None,
        "runtime_seconds": stats(item["duration"] for item in items),
    }


def task_groups(tasks):
    grouped = defaultdict(list)
    for task in tasks:
        details = task["details"]
        mode = details.get("segmentation_mode", "unknown")
        if not isinstance(mode, str) or mode not in MODES:
            mode = "unknown"
        key = (
            task["max_overlap"], str(details.get("slice_part") or "unknown"),
            str(details.get("model_id") or "unknown"), mode,
            str(details.get("study_type") or "unknown"),
        )
        grouped[key].append(task)
    result = []
    for key, items in sorted(grouped.items(), key=lambda pair: (
        pair[0][0] is None, pair[0][0] or 0, *pair[0][1:],
    )):
        stages = defaultdict(list)
        for item in items:
            values = item["details"].get("stage_seconds", {})
            if isinstance(values, dict):
                for name, value in values.items():
                    stages[name].append(number(value, nonnegative=True))
        result.append({
            "actual_max_overlap_during_task": key[0], "slice_part": key[1],
            "model_id": key[2], "segmentation_mode": key[3], "study_type": key[4],
            **interval_summary(items),
            "series_count": len({item["details"]["series_key"] for item in items
                                 if isinstance(item["details"].get("series_key"), str)}),
            "cache_wait_seconds": stats(number(item["details"].get("cache_wait_seconds"),
                                               nonnegative=True) for item in items),
            "stage_seconds": {name: stats(values) for name, values in sorted(stages.items())},
        })
    return result


def resource_groups(host_root, task_timelines, inputs, issues):
    buckets = {}
    seen_host, seen_gpu = set(), set()

    def bucket(boot, timestamp):
        overlap = task_timelines[boot].at(timestamp) if boot in task_timelines else None
        if overlap not in buckets:
            buckets[overlap] = {
                "actual_recorded_task_overlap": overlap, "host_sample_count": 0,
                "gpu_unique_query_count": 0, "cpu": [], "ram_available": [],
                "ram_unavailable": [], "gpu_devices": {},
                "psi_cpu": [], "psi_memory": [], "psi_io": [],
            }
        return buckets[overlap]

    if host_root is not None:
        for path in sorted(host_root.glob("host-*.jsonl")):
            inputs["host_log_files"] += 1
            try:
                with path.open(encoding="utf-8", errors="replace") as stream:
                    for line in stream:
                        try:
                            record = json.loads(line)
                            if not isinstance(record, dict) or record.get("schema_version") != 1:
                                raise ValueError("unsupported host record")
                            timestamp = number(record.get("monotonic_s"), nonnegative=True)
                            if timestamp is None:
                                raise ValueError("invalid host time")
                        except (ValueError, TypeError):
                            issues["invalid_host_lines"] += 1
                            continue
                        boot = boot_key(record)
                        if boot is not None:
                            identity = (boot, timestamp)
                            if identity in seen_host:
                                issues["duplicate_host_samples"] += 1
                                continue
                            seen_host.add(identity)
                        inputs["host_samples"] += 1
                        target = bucket(boot, timestamp)
                        target["host_sample_count"] += 1
                        cpu = record.get("cpu") or {}
                        memory = record.get("memory") or {}
                        if not isinstance(cpu, dict):
                            cpu = {}
                        if not isinstance(memory, dict):
                            memory = {}
                        target["cpu"].append(number(cpu.get("utilization_percent"), nonnegative=True))
                        available = number(memory.get("available_bytes"), nonnegative=True)
                        total = number(memory.get("total_bytes"), nonnegative=True)
                        target["ram_available"].append(available / MIB if available is not None else None)
                        unavailable = total - available if total is not None and available is not None else None
                        target["ram_unavailable"].append(
                            unavailable / MIB if unavailable is not None and unavailable >= 0 else None
                        )
                        pressure = record.get("pressure")
                        for resource in ("cpu", "memory", "io"):
                            try:
                                value = number(pressure[resource]["some"]["avg10"], nonnegative=True)
                            except (TypeError, KeyError):
                                value = None
                            target["psi_" + resource].append(value)
                        gpu = record.get("gpu")
                        if not isinstance(gpu, dict):
                            continue
                        sampled = number(gpu.get("sampled_monotonic_s"), nonnegative=True)
                        if sampled is None or sampled > timestamp:
                            issues["invalid_gpu_sample_times"] += 1
                            continue
                        identity = (boot or str(path), sampled)
                        if identity in seen_gpu:
                            continue
                        seen_gpu.add(identity)
                        target = bucket(boot, sampled)
                        target["gpu_unique_query_count"] += 1
                        devices = gpu.get("devices")
                        if not isinstance(devices, list):
                            issues["gpu_queries_without_devices"] += 1
                            continue
                        for device in devices:
                            if not isinstance(device, dict):
                                continue
                            index = str(device.get("index", "unknown"))
                            values = target["gpu_devices"].setdefault(
                                index, {"used": [], "total": [], "utilization": []},
                            )
                            for name, field in (("used", "memory_used_mib"), ("total", "memory_total_mib"),
                                                ("utilization", "utilization_percent")):
                                values[name].append(number(device.get(field), nonnegative=True))
            except OSError:
                issues["unreadable_host_logs"] += 1
    result = []
    for overlap, values in sorted(buckets.items(), key=lambda pair: (pair[0] is None, pair[0] or 0)):
        result.append({
            "actual_recorded_task_overlap": overlap,
            "host_sample_count": values["host_sample_count"],
            "sampled_cpu_utilization_percent": stats(values["cpu"]),
            "sampled_ram_available_mib": stats(values["ram_available"]),
            "sampled_ram_unavailable_mib": stats(values["ram_unavailable"]),
            "sampled_psi_some_avg10": {name: stats(values["psi_" + name])
                                      for name in ("cpu", "memory", "io")},
            "gpu_unique_query_count": values["gpu_unique_query_count"],
            "gpu_devices": [{
                "index": index,
                "sampled_memory_used_mib": stats(metrics["used"]),
                "sampled_memory_total_mib": stats(metrics["total"]),
                "sampled_utilization_percent": stats(metrics["utilization"]),
            } for index, metrics in sorted(values["gpu_devices"].items())],
        })
    return result


def build_report(tasks_root: Path, host_root: Path | None = None) -> dict:
    tasks_root = Path(tasks_root)
    host_root = Path(host_root) if host_root is not None else None
    if not tasks_root.is_dir() or host_root is not None and not host_root.is_dir():
        raise ValueError("input roots must be existing directories")
    issues, inputs = Counter(), Counter()
    events = read_events(tasks_root, issues, inputs)
    tasks = pair_intervals(events, "task", issues)
    attempts = pair_intervals(events, "inference", issues)
    task_times, inference_times = timelines(tasks), timelines(attempts)
    boot_summaries = []
    for boot in sorted(task_times):
        timeline = task_times[boot]
        duration = timeline.times[-1] - timeline.times[0]
        ended = sum(item["valid_interval"] and item["boot_id"] == boot for item in tasks)
        boot_summaries.append({
            "boot_id": boot, "recorded_window_seconds": round(duration, 6),
            "completed_per_hour_in_recorded_window": round(ended * 3600 / duration, 6)
            if duration > 0 else None,
            "max_task_overlap": max(timeline.counts, default=0),
            "task_exposure_seconds_by_overlap": timeline.exposure(),
            "max_inference_overlap": max(inference_times[boot].counts, default=0)
            if boot in inference_times else None,
        })
    resources = resource_groups(host_root, task_times, inputs, issues)
    inference_peaks = [item["max_overlap"] for item in attempts if item["max_overlap"] is not None]
    return {
        "schema_version": 1, "report_type": "recorded_capacity_observations",
        "methodology": {
            "scope": "Recorded Pipeline.run intervals; excludes CLI imports/startup and dispatcher queue time.",
            "overlap": "Half-open [start,end), within matching boot_id; task groups use the maximum encountered during that task.",
            "quantiles": "Linear interpolation of observed ended-task durations; always inspect sample counts.",
            "completed_count": "Tasks with task_end, including succeeded/failed/aborted; success is counted separately.",
            "resources": "CPU/RAM use host sample time; GPU uses deduplicated sampled_monotonic_s, not cache read time.",
            "queue_data": "Server queue data was not collected; queue latency is unavailable. Configured caps never define recorded overlap.",
        },
        "limitations": [
            "Only debug-enabled, present records are included. Missing/unfinished tasks make overlap a recorded lower bound.",
            "Resource maxima/minima are sampled observations, not guaranteed peaks; missing data is never replaced with zero.",
            "A task's brief maximum overlap does not mean it ran at that overlap throughout; exposure seconds are separate.",
            "This report does not infer a safe concurrency limit. Natural business without enough load does not establish capacity.",
        ],
        "inputs": {name: inputs[name] for name in (
            "task_log_files", "capacity_lines", "host_log_files", "host_samples",
        )},
        "issues": dict(sorted((key, value) for key, value in issues.items() if value)),
        "data_availability": {
            "task_intervals_with_comparable_boot": sum(item["max_overlap"] is not None for item in tasks),
            "tasks_without_comparable_intervals": sum(item["max_overlap"] is None for item in tasks),
            "host_samples_present": inputs["host_samples"] > 0,
            "gpu_unique_queries": sum(item["gpu_unique_query_count"] for item in resources),
        },
        "tasks": interval_summary(tasks),
        "task_groups": task_groups(tasks),
        "inference": {
            "attempt_count": len(attempts),
            "paired_interval_count": sum(item["valid_interval"] for item in attempts),
            "missing_start_count": sum(not item["has_start"] for item in attempts),
            "missing_end_count": sum(not item["has_end"] for item in attempts),
            "max_observed_overlap": max(inference_peaks) if inference_peaks else None,
            "runtime_seconds": stats(item["duration"] for item in attempts),
            "outcomes": dict(sorted(Counter(str(item["details"].get("outcome") or "unknown")
                                            for item in attempts if item["has_end"]).items())),
        },
        "recorded_boot_windows": boot_summaries,
        "resources_by_actual_overlap": resources,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-root", required=True, type=Path)
    parser.add_argument("--host-root", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="explicit JSON report destination")
    args = parser.parse_args(argv)
    try:
        report = build_report(args.tasks_root, args.host_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
    except (OSError, ValueError) as error:
        print(f"Capacity summary failed: {type(error).__name__}", file=sys.stderr)
        return 1
    tasks = report["tasks"]
    print(f"Recorded tasks={tasks['sample_count']}, ended={tasks['completed_count']}, "
          f"succeeded={tasks['succeeded_count']}, failed={tasks['failed_count']}, "
          f"max recorded overlap={tasks['max_observed_overlap']}, "
          f"host samples={report['inputs']['host_samples']}.")
    print("Sampled observations only; safe concurrency limit is not inferred. Queue time was not collected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
