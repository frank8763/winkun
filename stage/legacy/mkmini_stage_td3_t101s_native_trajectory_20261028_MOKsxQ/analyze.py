from pathlib import Path
import os
import csv
import json
import math
import traceback

ROOT = Path(os.environ["T101S_JOB"])

EXPECTED_SEEDS = {
    2029653601,
    2029653602,
    2029654801,
    2029654802,
}

def load(label, seed):
    case = ROOT / label
    identity = json.loads(
        (case / "identity.json").read_text()
    )

    log_path = case / "evaluate.log"

    if not log_path.is_file():
        raise RuntimeError(
            "EVALUATION_LOG_MISSING:" + label
        )

    lines = log_path.read_text(
        errors="replace"
    ).splitlines()

    if (
        "T71E_PAIR_GATE=PASS" not in lines
        or "T71E_STATE=EVALUATED" not in lines
    ):
        raise RuntimeError(
            "NATIVE_PAIR_GATE_FAILED:" + label
        )

    candidate = []
    reference = []

    for line in lines:
        if not line.startswith("T71E_CASE "):
            continue

        record = json.loads(
            line[len("T71E_CASE "):]
        )

        if record.get("model") == label:
            candidate.append(record)

        if record.get("model") == "T53":
            reference.append(record)

    if (
        len(candidate) != 4
        or len(reference) != 4
        or {
            r["actual_seed"] for r in candidate
        } != EXPECTED_SEEDS
        or {
            r["actual_seed"] for r in reference
        } != EXPECTED_SEEDS
    ):
        raise RuntimeError(
            "NATIVE_FOUR_TASK_EVIDENCE_INVALID:" + label
        )

    for record in candidate:
        if (
            record["checkpoint_sha256"]
            != identity["checkpoint_sha256"]
            or record["training"] is not False
            or record["action_override"] is not False
        ):
            raise RuntimeError(
                "MODEL_IDENTITY_INVALID:" + label
            )

    trace_path = (
        case / "traces"
        / f"{label}_seed{seed}.csv"
    )

    if not trace_path.is_file():
        raise RuntimeError(
            "TRACE_FILE_MISSING:" + str(trace_path)
        )

    records = {}

    with trace_path.open(newline="") as f:
        for row in csv.DictReader(f):
            step = int(row["step"])

            records[step] = {
                "step": step,
                "x": float(row["x_m"]),
                "y": float(row["y_m"]),
                "yaw": float(row["yaw_rad"]),
                "dist": float(row["goal_dist_m"]),
                "angle": float(row["goal_angle_rad"]),
                "steer": float(row["steer"]),
                "throttle": float(row["throttle"]),
                "speed": float(row["speed_mps"]),
                "lidar": float(row["lidar_min_m"]),
            }

    if len(records) < 100:
        raise RuntimeError(
            "TRACE_TOO_SHORT:" + label
        )

    print(
        "T101S_TRACE_VALID=PASS",
        "MODEL=", label,
        "SEED=", seed,
        "SAMPLES=", len(records)
    )

    return records

def angular_difference_deg(a, b):
    return math.degrees(
        math.atan2(
            math.sin(b - a),
            math.cos(b - a)
        )
    )

def compare(seed, title):

    print("\n" + "=" * 80)
    print("T101S_SCENE=", title)
    print("SEED=", seed)
    print("=" * 80)

    a = load("T87A", seed)
    b = load("T96A", seed)

    common = sorted(set(a) & set(b))

    if len(common) < 100:
        raise RuntimeError(
            "INSUFFICIENT_SYNCHRONIZED_STEPS"
        )

    first_position = None
    first_heading = None
    first_distance = None

    for step in common:

        x = a[step]
        y = b[step]

        pos_difference = math.hypot(
            y["x"] - x["x"],
            y["y"] - x["y"]
        )

        yaw_difference = abs(
            angular_difference_deg(
                x["yaw"], y["yaw"]
            )
        )

        dist_difference = abs(
            y["dist"] - x["dist"]
        )

        if (
            first_position is None
            and pos_difference >= 0.10
        ):
            first_position = step

        if (
            first_heading is None
            and yaw_difference >= 5.0
        ):
            first_heading = step

        if (
            first_distance is None
            and dist_difference >= 0.15
        ):
            first_distance = step

    print(
        "FIRST_POSITION_DIVERGENCE_0P10M=",
        first_position
    )

    print(
        "FIRST_HEADING_DIVERGENCE_5DEG=",
        first_heading
    )

    print(
        "FIRST_DISTANCE_DIVERGENCE_0P15M=",
        first_distance
    )

    print("\n========== 关键轨迹点 ==========")

    targets = [
        1, 1200, 2400, 3600, 4800,
        6000, 7200, 8400, 9600,
        10800, 12000, 13200
    ]

    for target in targets:

        closest = min(
            common,
            key=lambda step: abs(step - target)
        )

        if abs(closest - target) > 60:
            continue

        x = a[closest]
        y = b[closest]

        pos_difference = math.hypot(
            y["x"] - x["x"],
            y["y"] - x["y"]
        )

        yaw_difference = angular_difference_deg(
            x["yaw"], y["yaw"]
        )

        print(
            "STEP=", closest,
            "POS_DIFF_M=", round(pos_difference, 4),
            "YAW_DIFF_DEG=", round(yaw_difference, 3),
            "T87_DIST=", round(x["dist"], 4),
            "T96_DIST=", round(y["dist"], 4),
            "T87_STEER=", round(x["steer"], 4),
            "T96_STEER=", round(y["steer"], 4),
            "T87_THROTTLE=", round(x["throttle"], 4),
            "T96_THROTTLE=", round(y["throttle"], 4),
            "T87_SPEED=", round(x["speed"], 4),
            "T96_SPEED=", round(y["speed"], 4),
            "T87_LIDAR=", round(x["lidar"], 4),
            "T96_LIDAR=", round(y["lidar"], 4),
        )

    return {
        "seed": seed,
        "first_position_divergence": first_position,
        "first_heading_divergence": first_heading,
        "first_distance_divergence": first_distance,
        "common_samples": len(common),
    }

try:
    reports = []

    reports.append(
        compare(
            2029653601,
            "LEFT_MINUS_17P3"
        )
    )

    reports.append(
        compare(
            2029654802,
            "RIGHT_PLUS_18P8"
        )
    )

    (ROOT / "divergence_report.json").write_text(
        json.dumps(
            reports, indent=2
        ) + "\n"
    )

    print("\nT101S_TRAJECTORY_COMPARISON=PASS")

except Exception as error:
    print(
        "T101S_TRAJECTORY_COMPARISON_BLOCKED=",
        repr(error)
    )
    traceback.print_exc()

finally:
    print("TRAINING_LAUNCHED=FALSE")
    print("MODEL_PROMOTED=FALSE")
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
