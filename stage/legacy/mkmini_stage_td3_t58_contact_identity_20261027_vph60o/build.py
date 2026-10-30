from pathlib import Path
import ast
import hashlib
import json
import sys
import traceback

SRC = Path(sys.argv[1])
JOB = Path(sys.argv[2])

TASKS = [
    (2029654802, 18.8, 3.0)
]

EXPECTED = {
    "T53": (
        Path.home()
        / "mkmini_stage_td3_t53_actor_anchor_20261027_v1"
        / "checkpoints/episode_0010.pt",
        "8d4ad4008a7647dbe96ad67b66f0a22c0d51c035744d25259560d1e936e4e8e0"
    ),
    "T55": (
        Path.home()
        / "mkmini_stage_td3_t55_neargoal_reward_20261027_v1"
        / "checkpoints/episode_0015.pt",
        "e093967ce1191e0ca1a3d72e15a20338413e9e4d308d90ba99ad0fccb133bbac"
    ),
}


def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(block)

    return h.hexdigest()


def check(ok, message):
    if not ok:
        raise RuntimeError(message)


def patch_constants(source):
    changes = {
        "JOB": f"JOB = Path({str(JOB)!r})",
        "TASKS": f"TASKS = {TASKS!r}",
        "WALL": "WALL = 2400",
    }

    tree = ast.parse(source)
    found = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id in changes
            ):
                check(
                    target.id not in found,
                    "DUPLICATE:" + target.id
                )
                found[target.id] = node

    check(
        set(found) == set(changes),
        "MISSING_CONSTANTS:"
        + repr(set(changes) - set(found))
    )

    lines = source.splitlines(keepends=True)

    for name, node in sorted(
        found.items(),
        key=lambda item: item[1].lineno,
        reverse=True
    ):
        lines[
            node.lineno - 1:node.end_lineno
        ] = [changes[name] + "\n"]

    result = "".join(lines)
    ast.parse(result)

    return result


DETAILS = '''
def _t58_describe_geom(_gid):
    _bid = int(model.geom_bodyid[_gid])
    _original_body = _bid
    _chain = []
    _seen = set()

    while _bid not in _seen:
        _seen.add(_bid)

        _body_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            _bid
        )

        _chain.append({
            "body_id": _bid,
            "body_name": _body_name,
            "joint_count": int(
                model.body_jntnum[_bid]
            ),
        })

        if _bid == 0:
            break

        _next = int(
            model.body_parentid[_bid]
        )

        if _next == _bid:
            break

        _bid = _next

    return {
        "geom_id": _gid,
        "geom_name": mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            _gid
        ),
        "body_id": _original_body,
        "body_chain": _chain,
        "geom_type": int(
            model.geom_type[_gid]
        ),
        "geom_group": int(
            model.geom_group[_gid]
        ),
        "contype": int(
            model.geom_contype[_gid]
        ),
        "conaffinity": int(
            model.geom_conaffinity[_gid]
        ),
        "geom_world_xyz": (
            data.geom_xpos[_gid].tolist()
        ),
    }


_t58_details = []

for _idx in range(int(data.ncon)):
    _contact = data.contact[_idx]

    _g1 = int(_contact.geom1)
    _g2 = int(_contact.geom2)

    _t58_details.append({
        "contact_index": _idx,
        "geom1": _t58_describe_geom(_g1),
        "geom2": _t58_describe_geom(_g2),
        "penetration_m": float(
            _contact.dist
        ),
        "contact_world_xyz": (
            _contact.pos.tolist()
        ),
    })

print(
    "T58_CONTACT_IDENTITIES",
    json.dumps(
        _t58_details,
        ensure_ascii=False
    ),
    flush=True
)

_t58_joints = []

for _jid in range(int(model.njnt)):
    _jname = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        _jid
    )

    _lower = (_jname or "").lower()

    if any(
        word in _lower
        for word in (
            "steer",
            "front",
            "ackermann",
        )
    ):
        _qa = int(
            model.jnt_qposadr[_jid]
        )
        _va = int(
            model.jnt_dofadr[_jid]
        )

        _t58_joints.append({
            "joint_id": _jid,
            "name": _jname,
            "qpos": float(
                data.qpos[_qa]
            ),
            "qvel": float(
                data.qvel[_va]
            ),
        })

print(
    "T58_STEERING_JOINTS",
    json.dumps(
        _t58_joints,
        ensure_ascii=False
    ),
    flush=True
)
'''


def inject_geometry_details(source):
    tree = ast.parse(source)

    anchors = []

    for node in ast.walk(tree):

        if not isinstance(node, ast.Expr):
            continue

        call = node.value

        if not isinstance(call, ast.Call):
            continue

        if not isinstance(
            call.func,
            ast.Attribute
        ):
            continue

        if (
            isinstance(
                call.func.value,
                ast.Name
            )
            and call.func.value.id
                == "contacts"
            and call.func.attr == "sort"
        ):
            anchors.append(node)

    check(
        len(anchors) == 1,
        "CONTACT_SORT_ANCHOR_INVALID"
    )

    node = anchors[0]

    lines = source.splitlines(
        keepends=True
    )

    original_line = lines[
        node.lineno - 1
    ]

    indentation = original_line[
        :len(original_line)
        - len(original_line.lstrip())
    ]

    block = "".join(
        indentation + line + "\n"
        if line.strip()
        else "\n"
        for line in DETAILS.strip().splitlines()
    )

    lines.insert(
        node.end_lineno,
        block
    )

    result = "".join(lines)

    ast.parse(result)

    check(
        result.count(
            "T58_CONTACT_IDENTITIES"
        ) == 1,
        "GEOMETRY_PATCH_NOT_UNIQUE"
    )

    return result


def main():
    print("T58_STAGE_ONLY=TRUE")
    print("T58_TRAINING=FALSE")
    print("T58_PLANNER=FALSE")
    print("T58_ACTION_OVERRIDE=FALSE")
    print("T58_ENV_MODIFICATION=FALSE")
    print("T58_MODEL_WRITE=FALSE")

    original = SRC / "evaluate.py"
    setup_path = SRC / "t57r_setup.json"

    check(
        original.is_file()
        and setup_path.is_file(),
        "T57R_SOURCE_MISSING"
    )

    setup = json.loads(
        setup_path.read_text()
    )

    check(
        sha(original)
        == setup["repaired_sha256"],
        "T57R_SOURCE_CHANGED"
    )

    for name, (path, digest) in EXPECTED.items():

        check(
            path.is_file()
            and sha(path) == digest,
            "CHECKPOINT_CHANGED:" + name
        )

        check(
            digest in original.read_text(),
            "MODEL_NOT_IN_SOURCE:" + name
        )

        print(
            "T58_MODEL_VERIFIED=",
            name
        )

    result = patch_constants(
        original.read_text()
    )

    result = inject_geometry_details(
        result
    )

    target = JOB / "evaluate.py"

    check(
        not target.exists(),
        "TARGET_ALREADY_EXISTS"
    )

    target.write_text(result)

    meta = {
        "experiment": "STAGE_TD3_T58",
        "source_sha256": sha(original),
        "generated_sha256": sha(target),
        "target_seed": 2029654802,
        "target_angle_deg": 18.8,
        "target_distance_m": 3.0,
        "models": list(EXPECTED),
        "training": False,
        "automatic_promotion": False,
    }

    (
        JOB / "t58_setup.json"
    ).write_text(
        json.dumps(
            meta,
            indent=2
        ) + "\n"
    )

    print("T58_TARGETED_SCENES=1")
    print("T58_MUJOCO_RUNS=2")
    print("T58_READY=TRUE")


try:
    main()

except Exception as error:
    print(
        "T58_PREPARE=BLOCKED",
        repr(error)
    )
    traceback.print_exc()
    raise

finally:
    print(
        "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE"
    )
