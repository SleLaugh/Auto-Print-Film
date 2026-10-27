import importlib.util
import itertools
import json
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/collect_host_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("collect_host_diagnostics", SCRIPT)
monitor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor)


class HostDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.proc = self.base / "proc"
        self.proc.mkdir()
        self.output = self.base / "output"
        self.output.mkdir()
        self.sys = self.base / "sys"
        self.write("sys/kernel/random/boot_id", "example-boot\n")
        self.write(
            "stat",
            "cpu 100 0 50 800 50 0 0 0 100 0\ncpu0 100 0 0 400 0 0 0 0\ncpu1 0 0 50 400 50 0 0 0\n",
        )
        self.write(
            "meminfo",
            "MemTotal: 2000 kB\nMemAvailable: 1500 kB\n"
            "SwapTotal: 1000 kB\nSwapFree: 800 kB\nDirty: 12 kB\nWriteback: 3 kB\n",
        )
        self.write("loadavg", "0.10 0.20 0.30 1/100 101\n")
        self.write("diskstats", "")

    def write(self, relative, text):
        path = self.proc / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def process(self, pid, *, cpu=100, start=50, comm="python3", command=None,
                processor=None, ppid=7, state="S"):
        fields = ["0"] * (37 if processor is not None else 22)
        fields[0] = state
        fields[1] = str(ppid)
        fields[11] = str(cpu)
        fields[12] = "0"
        fields[17] = "4"
        fields[19] = str(start)
        fields[21] = "20"
        if processor is not None:
            fields[36] = str(processor)
        self.write(f"{pid}/stat", f"{pid} ({comm}) {' '.join(fields)}\n")
        (self.proc / str(pid) / "cmdline").write_bytes(
            command if command is not None else b"python3\0/path/patient-secret\0"
        )

    def sampler(self, **kwargs):
        return monitor.HostSampler(self.output, proc_root=self.proc, sys_root=self.sys, **kwargs)

    def test_deltas_memory_and_process_identity_without_command_line(self):
        self.process(10, comm="python worker )")
        self.process(11, comm="dotnet", command=b"dotnet\0/path/AutoPrintFilmService.dll\0")
        self.process(12, command=b"python\0nnUNetv2_predict\0/patient-secret\0")
        self.process(13, comm="sleep", command=b"sleep\010\0")
        sampler = self.sampler()
        timestamp = datetime(2026, 10, 26, tzinfo=timezone.utc)
        first = sampler.sample(100, timestamp)
        self.assertIsNone(first["cpu"]["utilization_percent"])
        self.assertEqual(first["cpu"]["total_ticks"], 1000)  # guest not counted twice
        self.assertEqual(first["memory"]["available_bytes"], 1500 * 1024)
        self.assertEqual(first["memory"]["swap_used_bytes"], 200 * 1024)
        self.assertEqual(first["memory"]["dirty_bytes"], 12 * 1024)
        self.assertEqual(first["memory"]["writeback_bytes"], 3 * 1024)
        self.assertEqual(first["kernel_release"], monitor.os.uname().release)
        self.assertIsNone(first["cpu"]["per_cpu"]["cpu0"]["utilization_percent"])
        self.assertEqual(
            [p["kind"] for p in first["processes"]], ["python", "service", "nnunet"]
        )
        self.assertNotIn("patient-secret", json.dumps(first))
        self.assertEqual(first["processes"][0]["rss_bytes"], 20 * sampler.page_size)
        self.assertIsNone(first["processes"][0]["processor"])
        self.write(
            "stat",
            "cpu 160 0 90 900 50 0 0 0 100 0\n"
            "cpu0 160 0 40 400 0 0 0 0\ncpu1 0 0 50 500 50 0 0 0\n",
        )
        self.process(10, cpu=100 + sampler.clock_ticks * 20, processor=1)
        second = sampler.sample(110, timestamp)
        self.assertEqual(second["cpu"]["utilization_percent"], 50.0)
        self.assertEqual(second["processes"][0]["cpu_percent"], 200.0)
        self.assertEqual(second["processes"][0]["processor"], 1)
        self.assertEqual(second["cpu"]["per_cpu"]["cpu0"]["utilization_percent"], 100)
        self.assertEqual(second["cpu"]["per_cpu"]["cpu1"]["utilization_percent"], 0)
        self.process(10, cpu=9000, start=999)  # same PID, different process
        self.assertIsNone(sampler.sample(120, timestamp)["processes"][0]["cpu_percent"])

    def test_optional_counters_and_a_failed_component_do_not_stop_sampling(self):
        self.write("pressure/memory", "some avg10=1.25 avg60=0.50 avg300=0.10 total=123\n")
        sensor = self.sys / "class/hwmon/hwmon0/temp1_input"
        sensor.parent.mkdir(parents=True)
        sensor.write_text("42000\n")
        self.write("stat", "malformed\n")
        self.process(15)
        self.write("16/stat", "broken process\n")
        record = self.sampler().sample(100, datetime.now(timezone.utc))
        self.assertIsNone(record["cpu"])
        self.assertEqual(record["errors"][0]["source"], "cpu")
        self.assertEqual(record["pressure"]["memory"]["some"]["total"], 123)
        self.assertNotIn("cpu", record["pressure"])
        self.assertEqual(record["temperatures"][0]["celsius"], 42.0)
        self.assertEqual(len(record["processes"]), 1)
        self.assertEqual(record["boot_id"], "example-boot")

    def test_gpu_is_opt_in_throttled_and_timeout_is_a_recorded_error(self):
        with mock.patch.object(monitor.subprocess, "run") as run:
            self.assertIsNone(self.sampler().gpu(100))
            run.assert_not_called()
            sampler = self.sampler(gpu=True)
            run.return_value = subprocess.CompletedProcess([], 0, "0, 20, 100, 200, N/A, 50\n", "")
            self.assertEqual(sampler.gpu(100)["devices"][0]["memory_used_mib"], 100)
            cached = sampler.gpu(129)
            self.assertEqual(cached["age_s"], 29)
            self.assertIsNone(cached["devices"][0]["power_watts"])
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.kwargs["timeout"], 2)
            run.side_effect = subprocess.TimeoutExpired("nvidia-smi", 2)
            failed = sampler.gpu(130)
            self.assertEqual(failed["error"]["type"], "TimeoutExpired")
            sampler.gpu(131)
            self.assertEqual(run.call_count, 2)

    def test_gpu_custom_interval_keeps_global_counters_and_retries_after_error(self):
        with mock.patch.object(monitor.subprocess, "run") as run:
            sampler = self.sampler(gpu=True, gpu_interval=5)
            run.return_value = subprocess.CompletedProcess(
                [], 0, "0, 20, 100, 200, 45, 50\n1, 30, 300, 400, N/A, 60\n", "",
            )
            first = sampler.gpu(100)
            self.assertEqual(len(first["devices"]), 2)
            self.assertEqual(first["poll_interval_seconds"], 5)
            self.assertEqual(sampler.gpu(104)["age_s"], 4)
            self.assertEqual(run.call_count, 1)
            run.side_effect = FileNotFoundError("nvidia-smi unavailable")
            self.assertIsNone(sampler.gpu(105)["devices"])
            self.assertEqual(sampler.gpu(109)["error"]["type"], "FileNotFoundError")
            self.assertEqual(run.call_count, 2)
            run.side_effect = None
            self.assertEqual(sampler.gpu(110)["devices"][1]["memory_used_mib"], 300)
            self.assertEqual(run.call_count, 3)
            command = run.call_args.args[0]
            self.assertIn("--query-gpu=", command[1])
            self.assertNotIn("--query-compute-apps", " ".join(command))

    def test_disk_io_rates_use_elapsed_time_and_reset_after_counters_or_identity_change(self):
        partition = self.sys / "dev/block/8:1/partition"
        partition.parent.mkdir(parents=True)
        partition.write_text("1\n")
        self.write("diskstats", "8 0 sda 10 0 100 20 20 0 200 40 0 50 80\n"
                   "8 1 sda1 10 0 100 20 20 0 200 40 0 50 80\n"
                   "7 0 loop0 10 0 100 20 20 0 200 40 0 50 80\n")
        sampler = self.sampler()
        first = sampler.disk_io(100)
        self.assertEqual([item["name"] for item in first["devices"]], ["sda"])
        self.assertIsNone(first["devices"][0]["delta"])
        self.write("diskstats", "8 0 sda 20 0 200 70 40 0 400 140 2 550 2080\n")
        item = sampler.disk_io(105)["devices"][0]
        self.assertEqual(item["interval_seconds"], 5)
        self.assertEqual(item["in_flight"], 2)
        self.assertEqual(item["delta"]["read_bytes_per_second"], 10240)
        self.assertEqual(item["delta"]["write_bytes_per_second"], 20480)
        self.assertEqual(item["delta"]["read_iops"], 2)
        self.assertEqual(item["delta"]["write_iops"], 4)
        self.assertEqual(item["delta"]["read_await_ms"], 5)
        self.assertEqual(item["delta"]["write_await_ms"], 5)
        self.assertEqual(item["delta"]["io_busy_percent"], 10)
        self.assertEqual(item["delta"]["average_queue_depth"], 0.4)
        self.write("diskstats", "8 0 sda 1 0 1 1 1 0 1 1 0 1 1\n")
        reset = sampler.disk_io(110)["devices"][0]
        self.assertEqual(reset["delta_status"], "counter_reset")
        self.assertIsNone(reset["delta"])
        self.write("diskstats", "8 0 sdb 2 0 2 2 2 0 2 2 0 2 2\n")
        changed = sampler.disk_io(115)["devices"][0]
        self.assertEqual(changed["delta_status"], "initial")
        self.assertIsNone(changed["delta"])

    def test_process_entry_points_do_not_treat_repo_paths_tests_or_collectors_as_tasks(self):
        cases = [
            (b"/repo/AutoPrintFilm/.venv/bin/python\0-m\0pytest\0", None),
            (b"python\0/repo/AutoPrintFilm/scripts/collect_host_diagnostics.py\0", None),
            (b"python\0/repo/AutoPrintFilm/scripts/capture_kernel_evidence.py\0", None),
            (b"python\0/repo/AutoPrintFilm/tests/test_main.py\0", None),
            (b"python\0-c\0print('/repo/AutoPrintFilm')\0", "python"),
            (b"python\0/repo/AutoPrintFilm/.venv/bin/nnUNetv2_predict\0", "nnunet"),
            (b"python\0/repo/AutoPrintFilm/src/inference/nnunet_export.py\0", "nnunet"),
            (b"python\0-c\0from nnunetv2.inference.predict_from_raw_data import "
             b"predict_entry_point; predict_entry_point()\0", "nnunet"),
            (b"python\0/repo/AutoPrintFilm/main.py\0-i\0/input\0-o\0/output\0", "autoprintfilm"),
            (b"python\0/other/main.py\0-i\0/input\0-o\0/output\0", "python"),
            (b"/opt/AutoPrintFilm\0--help\0", "auxiliary"),
            (b"/opt/AutoPrintFilm\0--multiprocessing-fork\0", "python"),
        ]
        for command, expected in cases:
            with self.subTest(command=command):
                self.assertEqual(monitor.process_kind("python3", command), expected)
        self.assertEqual(monitor.process_kind(
            "python3", b"python\0main.py\0-i\0/input\0-o\0/output\0",
            cwd="/repo/AutoPrintFilm",
        ), "autoprintfilm")

    def test_packaged_parent_resources_are_kept_with_launcher_role(self):
        command = b"/opt/AutoPrintFilm\0-i\0/private-input\0-o\0/private-output\0"
        self.process(10, comm="AutoPrintFilm", command=command)
        self.process(11, comm="AutoPrintFilm", command=command, ppid=10)
        self.process(12, comm="AutoPrintFilm", command=command, state="Z")
        self.process(13, command=b"python\0-m\0pytest\0/repo/AutoPrintFilm\0")
        self.process(14, command=b"python\0/opt/collect_host_diagnostics.py\0")
        self.process(99, comm="AutoPrintFilm", command=command)
        with mock.patch.object(monitor.os, "getpid", return_value=99):
            records = self.sampler().processes(100)
        self.assertEqual([item["pid"] for item in records], [10, 11, 12])
        self.assertEqual([item["role"] for item in records], ["launcher", "worker", "exited"])
        self.assertTrue(all(item["rss_bytes"] > 0 for item in records))
        self.assertNotIn("private-input", json.dumps(records))

    def test_disabled_missing_or_invalid_debug_config_never_constructs_sampler_or_writer(self):
        config = self.base / "logger.json"
        for text in (None, "{}", "{", "[]", '{"debug_log_file_enabled":false}',
                     '{"debug_log_file_enabled":"true"}', " " * 65537):
            with self.subTest(config=text[:40] if text else text):
                if text is None:
                    config.unlink(missing_ok=True)
                else:
                    config.write_text(text)
                output = self.base / "never-created"
                with mock.patch.object(monitor, "HostSampler") as sampler, \
                        mock.patch.object(monitor, "DailyJsonlWriter") as writer, \
                        mock.patch.object(monitor.subprocess, "run") as gpu:
                    sleep = mock.Mock()
                    monitor.collect(output, samples=2, gpu=True, logger_config=config, sleep=sleep)
                    sampler.assert_not_called()
                    writer.assert_not_called()
                    gpu.assert_not_called()
                    self.assertEqual(sleep.call_count, 1)
                self.assertFalse(output.exists())

    def test_debug_switch_reopens_with_fresh_cpu_and_gpu_baselines(self):
        config = self.base / "logger.json"
        config.write_text('{"debug_log_file_enabled":true}')
        self.process(10)
        self.write("diskstats", "8 0 sda 10 0 100 20 20 0 200 40 0 50 80\n")
        real_sampler = monitor.HostSampler
        created = []

        def factory(output, **kwargs):
            item = real_sampler(output, proc_root=self.proc, sys_root=self.sys, **kwargs)
            created.append(item)
            return item

        def toggle(_):
            config.write_text(json.dumps({"debug_log_file_enabled": len(created) == 1
                                         and not monitor.debug_logging_enabled(config)}))

        with mock.patch.object(monitor, "HostSampler", side_effect=factory), \
                mock.patch.object(monitor.subprocess, "run") as gpu:
            gpu.return_value = subprocess.CompletedProcess([], 0, "0, 20, 100, 200, 45, 50\n", "")
            monitor.collect(
                self.output, samples=3, interval=5, gpu=True, gpu_interval=5,
                logger_config=config, monotonic=mock.Mock(side_effect=itertools.count(100, 0.01)),
                sleep=toggle,
            )
        self.assertEqual(len(created), 2)
        self.assertEqual(gpu.call_count, 2)
        records = [json.loads(line) for line in next(self.output.glob("*.jsonl")).read_text().splitlines()]
        self.assertEqual(len(records), 2)
        self.assertTrue(all(item["cpu"]["utilization_percent"] is None for item in records))
        self.assertTrue(all(item["processes"][0]["cpu_percent"] is None for item in records))
        self.assertTrue(all(item["disk_io"]["devices"][0]["delta"] is None for item in records))
        self.assertTrue(all(item["gpu"]["age_s"] == 0 for item in records))

    def test_daily_writer_rotates_and_flushes_before_close(self):
        writer = monitor.DailyJsonlWriter(self.output)
        self.addCleanup(writer.close)
        writer.write({"timestamp_utc": "2026-10-26T23:59:59+00:00", "test": 1})
        first = self.output / "host-2026-10-26.jsonl"
        self.assertEqual(json.loads(first.read_text())["test"], 1)
        writer.write({"timestamp_utc": "2026-10-27T00:00:00+00:00", "test": 2})
        self.assertEqual(json.loads((self.output / "host-2026-10-27.jsonl").read_text())["test"], 2)

    def test_sampling_exception_continues_but_output_failure_propagates(self):
        sampler = mock.Mock(boot_id="test-boot", kernel_release="test-kernel")
        sampler.sample.side_effect = [
            RuntimeError("temporary reader failure"),
            {"timestamp_utc": "2026-10-26T00:00:00+00:00"},
        ]
        sleep = mock.Mock()
        monitor.collect(
            self.output,
            samples=2,
            sampler=sampler,
            sleep=sleep,
            now=lambda: datetime(2026, 10, 26, tzinfo=timezone.utc),
        )
        records = [
            json.loads(line) for line in next(self.output.glob("*.jsonl")).read_text().splitlines()
        ]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["errors"][0]["source"], "sample")
        self.assertEqual(sleep.call_count, 1)
        sampler.sample.side_effect = None
        sampler.sample.return_value = {"timestamp_utc": "2026-10-26T00:00:00+00:00"}
        with mock.patch.object(monitor.DailyJsonlWriter, "write", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                monitor.collect(self.output, samples=1, sampler=sampler)

    def test_cli_requires_output_and_rejects_invalid_limits(self):
        for args in (
            [],
            ["--output", str(self.output), "--samples", "0"],
            *(
                ["--output", str(self.output), "--interval", value]
                for value in ("0", "0.5", "nan", "inf")
            ),
            *(
                ["--output", str(self.output), "--gpu-interval", value]
                for value in ("0", "0.5", "nan", "inf", "-1")
            ),
        ):
            with self.subTest(args=args), mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                monitor.build_parser().parse_args(args)
        args = monitor.build_parser().parse_args(["--output", str(self.output)])
        self.assertEqual(args.interval, 10)
        self.assertIsNone(args.samples)
        self.assertFalse(args.gpu)
        self.assertEqual(args.gpu_interval, 30)
        self.assertIsNone(args.logger_config)
        configured = monitor.build_parser().parse_args([
            "--output", str(self.output), "--interval", "5", "--gpu", "--gpu-interval", "5",
            "--logger-config", str(self.base / "logger.json"),
        ])
        self.assertEqual(configured.gpu_interval, 5)
        self.assertEqual(configured.logger_config, self.base / "logger.json")


if __name__ == "__main__":
    unittest.main()
