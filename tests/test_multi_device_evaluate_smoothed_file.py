"""Tests for scoring a smoothed track file (beacon 0.51.0)."""

import contextlib
import io
import json
import os
import tempfile
import unittest

from beacon import (Beacon, MultiDeviceTracker,
                    evaluate_multi_device_track,
                    evaluate_smoothed_multi_device_track,
                    load_ground_truth, load_smoothed_track_file,
                    tracks_from_export_rows)
from beacon.export import (export_smoothed_multi_track_csv,
                           export_smoothed_multi_track_json,
                           export_smoothed_track_json,
                           export_track_json)
from beacon.multi_device import main as multi_main
from beacon.simulate import rectangular_walk, simulate_walk

# Four corner beacons in a 20 x 12 m ward, matching the CLI defaults.
BEACONS = [
    Beacon("A", 0.0, 0.0),
    Beacon("B", 20.0, 0.0),
    Beacon("C", 20.0, 12.0),
    Beacon("D", 0.0, 12.0),
]


def _two_walk_tracks_and_truths():
    """The multi-device demo tracks and truths, scored raw and smoothed.

    Tag-1 forward, seed 7; tag-2 the reverse walk, seed 11.
    """
    forward = simulate_walk(rectangular_walk(20.0, 12.0, 10), BEACONS,
                            noise_db=2.0, seed=7)
    reverse = simulate_walk(
        list(reversed(rectangular_walk(20.0, 12.0, 10))), BEACONS,
        noise_db=2.0, seed=11)
    tracker = MultiDeviceTracker(BEACONS)
    for i, step in enumerate(forward):
        tracker.feed("tag-1", step["rssi"], float(i))
    for i, step in enumerate(reverse):
        tracker.feed("tag-2", step["rssi"], float(i))
    truths = {
        "tag-1": [step["true"] for step in forward],
        "tag-2": [step["true"] for step in reverse],
    }
    return tracker.tracks(), truths


def _write_truth(truths, tmp, name="truth.json"):
    """Ground truth as the CLI reads it: one {t, x, y} point per
    step, t the step index, in step order."""
    path = os.path.join(tmp, name)
    data = {
        device_id: [{"t": float(i), "x": point[0], "y": point[1]}
                    for i, point in enumerate(points)]
        for device_id, points in truths.items()
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle)
    return path


class EvaluateSmoothedFileTest(unittest.TestCase):
    def test_file_report_matches_library_smoothed_report(self):
        # Scoring the smoothed export file gives the same numbers
        # as smoothing the raw tracks in code — the file path and
        # the library path cannot disagree about the smoothed
        # track.
        tracks, truths = _two_walk_tracks_and_truths()
        expected = evaluate_smoothed_multi_device_track(tracks, truths)
        with tempfile.TemporaryDirectory() as tmp:
            track_path = os.path.join(tmp, "smoothed.json")
            export_smoothed_multi_track_json(tracks, track_path)
            rows, method = load_smoothed_track_file(track_path)
            self.assertEqual(method, "rts")
            truth_path = _write_truth(truths, tmp)
            loaded = load_ground_truth(truth_path)
            file_tracks, scorer_truths = tracks_from_export_rows(
                rows, loaded)
            report = evaluate_multi_device_track(file_tracks,
                                                 scorer_truths)
        for device_id in ("tag-1", "tag-2"):
            self.assertAlmostEqual(
                report.by_device[device_id].mean_error_m,
                expected.by_device[device_id].mean_error_m, places=2)
        self.assertAlmostEqual(report.overall.mean_error_m, 1.25,
                               places=2)
        self.assertEqual(report.overall.steps, 80)
        self.assertEqual(report.overall.located, 80)

    def test_csv_file_scores_the_same(self):
        # The CSV export carries the same smoothed rows as the
        # JSON one, so it scores the same — the format is not a
        # second opinion.
        tracks, truths = _two_walk_tracks_and_truths()
        with tempfile.TemporaryDirectory() as tmp:
            track_path = os.path.join(tmp, "smoothed.csv")
            export_smoothed_multi_track_csv(tracks, track_path)
            rows, method = load_smoothed_track_file(track_path)
            self.assertEqual(method, "rts")
            truth_path = _write_truth(truths, tmp)
            file_tracks, scorer_truths = tracks_from_export_rows(
                rows, load_ground_truth(truth_path))
            report = evaluate_multi_device_track(file_tracks,
                                                 scorer_truths)
        self.assertAlmostEqual(
            report.by_device["tag-1"].mean_error_m, 1.45, places=2)
        self.assertAlmostEqual(
            report.by_device["tag-2"].mean_error_m, 1.04, places=2)
        self.assertAlmostEqual(report.overall.mean_error_m, 1.25,
                               places=2)

    def test_unlocated_step_keeps_coverage(self):
        # A truth step with no row stays unlocated in the file
        # report: the coverage is the track's, not the file's row
        # count, and the smoothed report never claims a step the
        # tracker never located.
        tracks, truths = _two_walk_tracks_and_truths()
        starved = {device_id: list(positions)
                   for device_id, positions in tracks.items()}
        starved["tag-1"][5] = None
        with tempfile.TemporaryDirectory() as tmp:
            track_path = os.path.join(tmp, "smoothed.json")
            export_smoothed_multi_track_json(starved, track_path)
            rows, _ = load_smoothed_track_file(track_path)
            truth_path = _write_truth(truths, tmp)
            file_tracks, scorer_truths = tracks_from_export_rows(
                rows, load_ground_truth(truth_path))
            report = evaluate_multi_device_track(file_tracks,
                                                 scorer_truths)
        self.assertEqual(report.by_device["tag-1"].steps, 40)
        self.assertEqual(report.by_device["tag-1"].located, 39)
        self.assertAlmostEqual(report.by_device["tag-1"].coverage,
                               39 / 40)
        self.assertEqual(report.overall.located, 79)

    def test_single_device_unnamed_rows(self):
        # A single-device export carries no device on its rows,
        # so it scores against the truth's single device without
        # the file having to name it.
        tracks, truths = _two_walk_tracks_and_truths()
        single_truth = {"tag-1": truths["tag-1"]}
        with tempfile.TemporaryDirectory() as tmp:
            track_path = os.path.join(tmp, "smoothed.json")
            export_smoothed_track_json(tracks["tag-1"], track_path)
            rows, _ = load_smoothed_track_file(track_path)
            self.assertIsNone(rows[0]["device"])
            truth_path = _write_truth(single_truth, tmp)
            file_tracks, scorer_truths = tracks_from_export_rows(
                rows, load_ground_truth(truth_path))
            report = evaluate_multi_device_track(file_tracks,
                                                 scorer_truths)
        self.assertAlmostEqual(
            report.by_device["tag-1"].mean_error_m, 1.45, places=2)
        self.assertEqual(report.overall.steps, 40)

    def test_raw_json_file_is_refused(self):
        # A raw export scored as a smoothed one would report the
        # live filter's numbers under the smoother's name, so the
        # JSON header is checked rather than trusted.
        tracks, _truths = _two_walk_tracks_and_truths()
        with tempfile.TemporaryDirectory() as tmp:
            track_path = os.path.join(tmp, "raw.json")
            export_track_json(tracks["tag-1"], track_path)
            with self.assertRaises(ValueError):
                load_smoothed_track_file(track_path)

    def test_unknown_extension_raises(self):
        with self.assertRaises(ValueError):
            load_smoothed_track_file("smoothed.txt")

    def test_row_without_truth_raises(self):
        # A row whose timestamp has no truth step cannot be
        # aligned, so the file and the truth are refused instead
        # of scored on a guessed alignment.
        tracks, truths = _two_walk_tracks_and_truths()
        with tempfile.TemporaryDirectory() as tmp:
            track_path = os.path.join(tmp, "smoothed.json")
            export_smoothed_multi_track_json(tracks, track_path)
            rows, _ = load_smoothed_track_file(track_path)
            short_truth = {
                device_id: [{"t": float(i), "x": point[0],
                             "y": point[1]}
                            for i, point in enumerate(points[:-1])]
                for device_id, points in truths.items()
            }
            truth_path = os.path.join(tmp, "short-truth.json")
            with open(truth_path, "w", encoding="utf-8") as handle:
                json.dump(short_truth, handle)
            with self.assertRaises(ValueError):
                tracks_from_export_rows(rows,
                                        load_ground_truth(truth_path))

    def test_truth_device_without_rows_raises(self):
        # A device the truth names but the track never had is a
        # disagreement between the file and the truth, not a
        # device that scored zero.
        tracks, truths = _two_walk_tracks_and_truths()
        with tempfile.TemporaryDirectory() as tmp:
            track_path = os.path.join(tmp, "smoothed.json")
            export_smoothed_multi_track_json(tracks, track_path)
            rows, _ = load_smoothed_track_file(track_path)
            truths["tag-3"] = truths["tag-1"]
            truth_path = _write_truth(truths, tmp)
            with self.assertRaises(ValueError):
                tracks_from_export_rows(rows,
                                        load_ground_truth(truth_path))

    def test_ground_truth_point_without_t_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            truth_path = os.path.join(tmp, "truth.json")
            with open(truth_path, "w", encoding="utf-8") as handle:
                json.dump({"tag-1": [{"x": 1.0, "y": 2.0}]}, handle)
            with self.assertRaises(ValueError):
                load_ground_truth(truth_path)

    def test_ground_truth_mixed_floor_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            truth_path = os.path.join(tmp, "truth.json")
            with open(truth_path, "w", encoding="utf-8") as handle:
                json.dump({"tag-1": [{"t": 0.0, "x": 1.0, "y": 2.0},
                                     {"t": 1.0, "x": 1.5, "y": 2.5,
                                      "floor": 1}]}, handle)
            with self.assertRaises(ValueError):
                load_ground_truth(truth_path)

    def test_ground_truth_with_floor_scores(self):
        # A multi-floor truth carries the floor per point, and
        # the report's floor accuracy comes from it.
        tracks, truths = _two_walk_tracks_and_truths()
        floored = {
            device_id: [{"t": float(i), "x": point[0], "y": point[1],
                         "floor": 0}
                        for i, point in enumerate(points)]
            for device_id, points in truths.items()
        }
        with tempfile.TemporaryDirectory() as tmp:
            track_path = os.path.join(tmp, "smoothed.json")
            export_smoothed_multi_track_json(tracks, track_path)
            rows, _ = load_smoothed_track_file(track_path)
            truth_path = os.path.join(tmp, "truth.json")
            with open(truth_path, "w", encoding="utf-8") as handle:
                json.dump(floored, handle)
            file_tracks, scorer_truths = tracks_from_export_rows(
                rows, load_ground_truth(truth_path))
            report = evaluate_multi_device_track(file_tracks,
                                                 scorer_truths)
        self.assertEqual(report.overall.floor_accuracy, 1.0)


class EvaluateSmoothedFileCliTest(unittest.TestCase):
    def _report_lines(self, output):
        return [line for line in output.splitlines()
                if line.startswith(("tag-1:", "tag-2:", "overall:"))]

    def test_cli_prints_file_report(self):
        tracks, truths = _two_walk_tracks_and_truths()
        with tempfile.TemporaryDirectory() as tmp:
            track_path = os.path.join(tmp, "smoothed.json")
            export_smoothed_multi_track_json(tracks, track_path)
            truth_path = _write_truth(truths, tmp)
            out = io.StringIO()
            self.assertEqual(
                multi_main(argv=["--evaluate-smoothed-file", track_path,
                                "--ground-truth", truth_path],
                           output_stream=out), 0)
        self.assertIn(
            f"Smoothed track file ({track_path}, rts):",
            out.getvalue())
        lines = self._report_lines(out.getvalue())
        self.assertEqual(len(lines), 3)
        self.assertIn("mean error: 1.45 m", lines[0])
        self.assertIn("mean error: 1.04 m", lines[1])
        self.assertIn("mean error: 1.25 m", lines[2])

    def test_cli_json_writes_report_dict(self):
        tracks, truths = _two_walk_tracks_and_truths()
        with tempfile.TemporaryDirectory() as tmp:
            track_path = os.path.join(tmp, "smoothed.json")
            export_smoothed_multi_track_json(tracks, track_path)
            truth_path = _write_truth(truths, tmp)
            report_path = os.path.join(tmp, "report.json")
            out = io.StringIO()
            self.assertEqual(
                multi_main(argv=["--evaluate-smoothed-file", track_path,
                                "--ground-truth", truth_path,
                                "--evaluate-smoothed-file-json",
                                report_path],
                           output_stream=out), 0)
            with open(report_path, encoding="utf-8") as handle:
                written = json.load(handle)
        self.assertTrue(written["smoothed"])
        self.assertEqual(written["smoothing_method"], "rts")
        self.assertEqual(set(written) - {"smoothed",
                                         "smoothing_method"},
                         {"by_device", "overall"})
        self.assertAlmostEqual(
            written["by_device"]["tag-1"]["mean_error_m"], 1.45,
            places=2)
        self.assertAlmostEqual(
            written["by_device"]["tag-2"]["mean_error_m"], 1.04,
            places=2)
        self.assertAlmostEqual(written["overall"]["mean_error_m"],
                               1.25, places=2)
        self.assertEqual(written["overall"]["steps"], 80)
        # The report prints as well as writing JSON: the file
        # flag names the track and prints its verdict.
        self.assertIn("mean error: 1.25 m", out.getvalue())

    def test_cli_csv_file_scores_the_same(self):
        tracks, truths = _two_walk_tracks_and_truths()
        with tempfile.TemporaryDirectory() as tmp:
            track_path = os.path.join(tmp, "smoothed.csv")
            export_smoothed_multi_track_csv(tracks, track_path)
            truth_path = _write_truth(truths, tmp)
            out = io.StringIO()
            self.assertEqual(
                multi_main(argv=["--evaluate-smoothed-file", track_path,
                                "--ground-truth", truth_path],
                           output_stream=out), 0)
        lines = self._report_lines(out.getvalue())
        self.assertEqual(len(lines), 3)
        self.assertIn("mean error: 1.25 m", lines[2])

    def test_cli_missing_file_returns_2(self):
        tracks, truths = _two_walk_tracks_and_truths()
        with tempfile.TemporaryDirectory() as tmp:
            truth_path = _write_truth(truths, tmp)
            out = io.StringIO()
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                result = multi_main(
                    argv=["--evaluate-smoothed-file",
                          os.path.join(tmp, "no-such-file.json"),
                          "--ground-truth", truth_path],
                    output_stream=out)
        self.assertEqual(result, 2)
        self.assertIn("no-such-file.json", err.getvalue())

    def test_file_flag_needs_ground_truth(self):
        with self.assertRaises(SystemExit) as caught:
            multi_main(argv=["--evaluate-smoothed-file",
                            "smoothed.json"],
                       output_stream=io.StringIO())
        self.assertEqual(caught.exception.code, 2)

    def test_ground_truth_needs_file_flag(self):
        with self.assertRaises(SystemExit) as caught:
            multi_main(argv=["--ground-truth", "truth.json"],
                       output_stream=io.StringIO())
        self.assertEqual(caught.exception.code, 2)

    def test_file_json_flag_needs_file_flag(self):
        with self.assertRaises(SystemExit) as caught:
            multi_main(argv=["--evaluate-smoothed-file-json",
                            "report.json",
                            "--ground-truth", "truth.json"],
                       output_stream=io.StringIO())
        self.assertEqual(caught.exception.code, 2)

    def test_file_flag_rejects_demo(self):
        with self.assertRaises(SystemExit) as caught:
            multi_main(argv=["--demo", "--evaluate-smoothed-file",
                            "smoothed.json",
                            "--ground-truth", "truth.json"],
                       output_stream=io.StringIO())
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
