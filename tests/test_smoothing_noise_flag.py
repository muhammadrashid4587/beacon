"""The --smoothing-noise-m CLI flag (beacon 0.55.0).

The smoother's measurement-noise value is a tunable library
parameter (``measurement_noise_m``, defaulting to the measured
2 m) — but a ward dashboard that only runs the CLIs had no way
to set it. The flag exposes it on every CLI that smooths: the
single-device demo summary (``beacon.cli``), the live tracker's
smoothed exports (``beacon.tracking``) and the multi-device
smoothed exports and smoothed demo reports
(``beacon.multi_device``) — so the overlay, the replay, the
export and the reports all honour the same value, and passing
1.0 reproduces the pre-0.54.0 numbers exactly. A non-positive
value is rejected by the parser instead of smoothing on a
made-up noise. A track file scored with
``--evaluate-smoothed-file`` is already smoothed, so the flag
does not re-smooth it.
"""

import contextlib
import io
import json
import re
import statistics
import tempfile
import unittest

from beacon.cli import main as cli_main
from beacon.export import load_track_csv, load_track_json
from beacon.multi_device import main as multi_main
from beacon.tracking import main as tracking_main


def _mean_semi_major(rows):
    return statistics.fmean(
        row["smoothed_ellipse_semi_major_m"] for row in rows)


def _smoothed_mean_error(output: str, label: str) -> float:
    """The mean error on the ``label`` line of a printed report."""
    for line in output.splitlines():
        if line.startswith(label):
            match = re.search(r"mean error: ([0-9.]+) m", line)
            assert match is not None, line
            return float(match.group(1))
    raise AssertionError(f"no {label!r} line in:\n{output}")


class CliSmoothingNoiseTest(unittest.TestCase):
    def _cli_output(self, argv):
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            self.assertEqual(cli_main(argv), 0)
        return sink.getvalue()

    def test_default_is_the_calibrated_noise(self):
        # No flag: the summary scores the calibrated smoothing —
        # the README's 1.34 m smoothed mean on the demo walk.
        output = self._cli_output([])
        self.assertAlmostEqual(
            _smoothed_mean_error(output, "smoothed:"), 1.34, places=2)

    def test_one_metre_reproduces_the_old_numbers(self):
        # 1.0 m is the pre-0.54.0 filter default: the smoothed
        # mean goes back to 1.45 m on the same walk.
        output = self._cli_output(["--smoothing-noise-m", "1.0"])
        self.assertAlmostEqual(
            _smoothed_mean_error(output, "smoothed:"), 1.45, places=2)

    def test_non_positive_noise_is_rejected(self):
        # A filter that trusts its measurements by a non-positive
        # amount is nonsense: the parser refuses it, exit code 2.
        for value in ("0", "-1.5"):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    cli_main(["--smoothing-noise-m", value])
            self.assertEqual(caught.exception.code, 2)


class TrackingSmoothingNoiseTest(unittest.TestCase):
    def _export_smoothed(self, argv):
        with tempfile.TemporaryDirectory() as tmp:
            json_path = f"{tmp}/track-smoothed.json"
            csv_path = f"{tmp}/track-smoothed.csv"
            sink = io.StringIO()
            with contextlib.redirect_stdout(sink):
                self.assertEqual(
                    tracking_main(argv + ["--export-smoothed-json",
                                          json_path,
                                          "--export-smoothed-csv",
                                          csv_path]),
                    0)
            return (load_track_json(json_path),
                    load_track_csv(csv_path))

    def test_default_export_is_the_calibrated_smoothing(self):
        # No flag: the smoothed export's ellipse averages 1.90 m
        # semi-major — the calibrated value, in both formats.
        rows_json, rows_csv = self._export_smoothed(["--demo"])
        self.assertAlmostEqual(_mean_semi_major(rows_json), 1.90,
                               places=2)
        self.assertAlmostEqual(_mean_semi_major(rows_csv), 1.90,
                               places=2)

    def test_one_metre_reproduces_the_old_ellipse(self):
        # 1.0 m: the ellipse goes back to the pre-0.54.0 1.14 m
        # mean semi-major, in both formats.
        rows_json, rows_csv = self._export_smoothed(
            ["--demo", "--smoothing-noise-m", "1.0"])
        self.assertAlmostEqual(_mean_semi_major(rows_json), 1.14,
                               places=2)
        self.assertAlmostEqual(_mean_semi_major(rows_csv), 1.14,
                               places=2)

    def test_non_positive_noise_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                tracking_main(["--demo", "--smoothing-noise-m", "0"])
        self.assertEqual(caught.exception.code, 2)


class MultiDeviceSmoothingNoiseTest(unittest.TestCase):
    def _multi_output(self, argv):
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            self.assertEqual(multi_main(argv), 0)
        return sink.getvalue()

    def test_default_smoothed_report_is_the_calibrated_one(self):
        # No flag: the smoothed demo report scores the calibrated
        # smoothing — the README's 1.13 m overall mean.
        output = self._multi_output(
            ["--demo", "--evaluate-smoothed"])
        self.assertAlmostEqual(
            _smoothed_mean_error(output, "overall:"), 1.13, places=2)

    def test_one_metre_reproduces_the_old_report(self):
        # 1.0 m: the overall smoothed mean goes back to the
        # pre-0.54.0 1.25 m.
        output = self._multi_output(
            ["--demo", "--evaluate-smoothed",
             "--smoothing-noise-m", "1.0"])
        self.assertAlmostEqual(
            _smoothed_mean_error(output, "overall:"), 1.25, places=2)

    def test_smoothed_export_honours_the_flag(self):
        # The combined smoothed export is smoothed with the
        # flag's value too, so the file the overlay and the replay
        # draw cannot disagree with the report.
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/multi-smoothed.json"
            self._multi_output(
                ["--demo", "--export-smoothed-json", path,
                 "--smoothing-noise-m", "1.0"])
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            self.assertTrue(data["smoothed"])
            self.assertAlmostEqual(_mean_semi_major(data["rows"]),
                                   1.14, places=2)

    def test_non_positive_noise_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                multi_main(["--demo", "--smoothing-noise-m", "-2"])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
