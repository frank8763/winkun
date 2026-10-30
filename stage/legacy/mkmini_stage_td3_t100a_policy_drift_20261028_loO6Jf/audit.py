from pathlib import Path
import os
import sys
import json
import hashlib
import inspect
import traceback

import numpy as np
import torch

H = Path.home()
OUT = Path(os.environ["T100A_JOB"])

T87 = H / (
    "mkmini_stage_td3_t87_direction_learning_"
    "20261028_Mnr36j"
)

T96 = H / (
    "mkmini_stage_td3_t96m_balanced_sampling_"
    "20261028_zgWvJE"
)

T95 = H / (
    "mkmini_stage_td3_t95_balanced_replay_"
    "20261028_aJuZyx"
)

T46 = H / (
    "mkmini_stage_td3_t42_verified_3m_replay_"
    "20261027_v1"
)

CORE = T87 / "td3_core_goal_contrast.py"

CHECKPOINTS = {
    "T87_BASE": (
        T87 / "checkpoints/episode_0025.pt",
        "0378e4af4296683d5d612a2d71607149e8fb3b0ff5235e35643766059ffaf94e",
    ),
    "T96_EP26": (
        T96 / "checkpoints/episode_0026.pt",
        "ce5400573123025140337656b33637104942d8befd2d1870fc397fcce2e7e6e0",
    ),
    "T96_EP27": (
        T96 / "checkpoints/episode_0027.pt",
        "f16169414c8a51aec84834e4c37836723de7a8dc706df60a02a276d8e8962646",
    ),
    "T96_EP28": (
        T96 / "checkpoints/episode_0028.pt",
        "97a2cc3d9999df2f658b2bdf8f8ccb33c98cbcb80ff63ef2fb140c6444e7d913",
    ),
}

ARCHIVE = T95 / "balanced_archive.pt"

SUCCESS = T46 / "verified_3m_success.npz"

EXPECTED_ARCHIVE_SHA = (
    "df09d48719ebe35ec726b09c3edd19b7fe861f55309b46ebd93be171604e68f6"
)

EXPECTED_SUCCESS_SHA = (
    "743c3b3099031a0f4104e2e41af22c12d491a70a6f3d780ca34477463cf9192b"
)

EXPECTED_CORE_SHA = (
    "59b3b27bce8c3eeefc4c1b12989139a8fbe00474cc2784df0d8f5a5ec4353a64"
)

def check(ok, message):
    if not ok:
        raise RuntimeError(message)

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def load_actor_class():

    check(
        CORE.is_file()
        and sha(CORE) == EXPECTED_CORE_SHA,
        "CORE_SOURCE_CHANGED"
    )

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "t100_readonly_td3_core",
        str(CORE)
    )

    check(
        spec is not None
        and spec.loader is not None,
        "ACTOR_IMPORT_SPEC_INVALID"
    )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    actor_class = getattr(module, "Actor", None)

    check(
        actor_class is not None,
        "ACTOR_CLASS_NOT_FOUND"
    )

    print(
        "ACTOR_SIGNATURE=",
        str(inspect.signature(actor_class))
    )

    return actor_class

def build_actor(actor_class, weights):

    errors = []

    for args in [
        (112, 2),
        (112, 2, 1.0),
        (112,),
        (),
    ]:

        try:
            model = actor_class(*args)

            model.load_state_dict(
                weights,
                strict=True
            )

            model.eval()

            with torch.no_grad():
                result = model(
                    torch.zeros(
                        (2, 112),
                        dtype=torch.float32
                    )
                )

            check(
                tuple(result.shape) == (2, 2),
                "ACTOR_OUTPUT_SHAPE_INVALID"
            )

            check(
                torch.isfinite(result).all().item(),
                "ACTOR_OUTPUT_NONFINITE"
            )

            check(
                torch.max(
                    torch.abs(result)
                ).item() <= 1.0001,
                "ACTOR_OUTPUT_RANGE_INVALID"
            )

            print(
                "ACTOR_CONSTRUCTOR_PASS=",
                args
            )

            return model

        except Exception as error:
            errors.append(
                repr(args) + ": " + repr(error)
            )

    for error in errors:
        print("ACTOR_CONSTRUCTOR_ATTEMPT=", error)

    raise RuntimeError(
        "ACTOR_CONSTRUCTION_BLOCKED"
    )

def predict(model, states):

    result = []

    with torch.no_grad():

        for start in range(
            0, len(states), 512
        ):

            batch = torch.from_numpy(
                np.ascontiguousarray(
                    states[start:start + 512],
                    dtype=np.float32
                )
            )

            action = model(batch)

            result.append(
                action.detach().cpu().numpy()
            )

    output = np.concatenate(
        result,
        axis=0
    )

    check(
        output.shape == (len(states), 2)
        and np.isfinite(output).all(),
        "ACTOR_BATCH_OUTPUT_INVALID"
    )

    return output

def limit(states, maximum=5000):

    check(
        len(states) > 0,
        "EMPTY_STATE_GROUP"
    )

    if len(states) > maximum:

        ids = np.linspace(
            0,
            len(states) - 1,
            maximum,
            dtype=np.int64
        )

        states = states[ids]

    return np.ascontiguousarray(
        states,
        dtype=np.float32
    )

def describe(label, reference, candidate):

    delta = candidate - reference

    steer_delta = np.abs(
        delta[:, 1]
    )

    throttle_delta = np.abs(
        delta[:, 0]
    )

    meaningful = (
        (np.abs(reference[:, 1]) >= 0.05)
        & (np.abs(candidate[:, 1]) >= 0.05)
    )

    sign_flip = (
        np.sign(reference[:, 1])
        != np.sign(candidate[:, 1])
    )

    flip_count = int(
        np.sum(sign_flip & meaningful)
    )

    comparable = int(
        np.sum(meaningful)
    )

    result = {
        "mean_abs_raw_steer_delta":
            float(np.mean(steer_delta)),
        "p95_abs_raw_steer_delta":
            float(np.percentile(steer_delta, 95)),
        "mean_abs_raw_throttle_delta":
            float(np.mean(throttle_delta)),
        "mean_reference_raw_throttle":
            float(np.mean(reference[:, 0])),
        "mean_candidate_raw_throttle":
            float(np.mean(candidate[:, 0])),
        "steer_sign_flips": flip_count,
        "comparable_steering_states":
            comparable,
        "steer_sign_flip_fraction":
            (
                flip_count / comparable
                if comparable else None
            ),
    }

    print(
        "POLICY_DRIFT",
        "GROUP=", label,
        "MEAN_STEER_DELTA=",
        round(
            result["mean_abs_raw_steer_delta"],
            5
        ),
        "P95_STEER_DELTA=",
        round(
            result["p95_abs_raw_steer_delta"],
            5
        ),
        "MEAN_THROTTLE_DELTA=",
        round(
            result["mean_abs_raw_throttle_delta"],
            5
        ),
        "REF_THROTTLE=",
        round(
            result["mean_reference_raw_throttle"],
            5
        ),
        "NEW_THROTTLE=",
        round(
            result["mean_candidate_raw_throttle"],
            5
        ),
        "STEER_SIGN_FLIPS=",
        f"{flip_count}/{comparable}",
        flush=True
    )

    return result

def archive_states(rows):

    states = np.stack([
        np.asarray(
            row[0],
            dtype=np.float32
        ).reshape(-1)
        for row in rows
    ])

    check(
        states.shape[1] == 112
        and np.isfinite(states).all(),
        "ARCHIVE_STATES_INVALID"
    )

    return states

try:
    print("=" * 78)
    print("T100A STAGE TD3 FROZEN-ACTOR BEHAVIOR DRIFT")
    print("=" * 78)

    print("TRAINING=FALSE")
    print("SIMULATION=FALSE")
    print("ACTION_OVERRIDE=FALSE")
    print("MODEL_WRITE=FALSE")
    print("HISTORICAL_MODEL_MODIFIED=FALSE")

    actor_class = load_actor_class()

    models = {}

    for label, (path, expected_sha) in CHECKPOINTS.items():

        check(
            path.is_file()
            and sha(path) == expected_sha,
            "CHECKPOINT_CHANGED:" + label
        )

        saved = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

        check(
            isinstance(saved.get("actor"), dict),
            "ACTOR_WEIGHTS_MISSING:" + label
        )

        models[label] = build_actor(
            actor_class,
            saved["actor"]
        )

        print(
            "MODEL_VERIFIED=",
            label,
            "FRAMES=", saved.get("frames"),
            "UPDATES=", saved.get("updates")
        )

        del saved

    check(
        ARCHIVE.is_file()
        and sha(ARCHIVE) == EXPECTED_ARCHIVE_SHA,
        "T95_ARCHIVE_CHANGED"
    )

    check(
        SUCCESS.is_file()
        and sha(SUCCESS) == EXPECTED_SUCCESS_SHA,
        "T46_SUCCESS_ARCHIVE_CHANGED"
    )

    datasets = {}

    with np.load(
        SUCCESS,
        allow_pickle=False
    ) as data:

        states = data["states"]
        sides = data["sides"]

        check(
            states.shape[1] == 112
            and len(sides) == len(states),
            "SUCCESS_STATES_INVALID"
        )

        distance = states[:, 48] * 10.0

        for side in ["LEFT", "RIGHT"]:

            selected = (
                sides == side
            )

            datasets[
                "SUCCESS_" + side
            ] = limit(
                states[selected]
            )

            near = (
                selected
                & (distance <= 1.0)
            )

            if np.any(near):

                datasets[
                    "SUCCESS_" + side + "_NEAR_GOAL"
                ] = limit(
                    states[near]
                )

    archived = torch.load(
        ARCHIVE,
        map_location="cpu",
        weights_only=False
    )

    datasets["DIFFICULT_LEFT"] = limit(
        archive_states(
            archived["LEFT"]
        )
    )

    datasets["DIFFICULT_RIGHT"] = limit(
        archive_states(
            archived["RIGHT"]
        )
    )

    print("\n" + "=" * 78)
    print("FROZEN STATE GROUPS")
    print("=" * 78)

    for name, states in datasets.items():
        print(
            "GROUP=", name,
            "SAMPLES=", len(states)
        )

    report = {}

    for group_name, states in datasets.items():

        print("\n" + "=" * 78)
        print("GROUP=", group_name)
        print("=" * 78)

        reference = predict(
            models["T87_BASE"],
            states
        )

        report[group_name] = {}

        for label in [
            "T96_EP26",
            "T96_EP27",
            "T96_EP28",
        ]:

            candidate = predict(
                models[label],
                states
            )

            report[group_name][label] = describe(
                group_name + "/" + label,
                reference,
                candidate
            )

    result_path = OUT / "policy_drift.json"

    result_path.write_text(
        json.dumps(
            report,
            indent=2
        ) + "\n"
    )

    print("\nT100_FIXED_OBSERVATIONS=PASS")
    print("T100_ACTOR_BEHAVIOR_AUDIT=PASS")
    print("RESULT_PATH=", result_path)

except Exception as error:

    print(
        "T100_AUDIT_BLOCKED=",
        repr(error)
    )

    traceback.print_exc()

finally:

    print("TRAINING_LAUNCHED=FALSE")
    print("SIMULATION_LAUNCHED=FALSE")
    print("MODEL_PROMOTED=FALSE")
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
