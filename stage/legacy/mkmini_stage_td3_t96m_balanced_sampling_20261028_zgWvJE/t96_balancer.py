
from pathlib import Path
import hashlib
import numpy as np
import torch

class T96BalancedSampler:

    def __init__(self, path, expected_sha):
        path = Path(path)

        h = hashlib.sha256(path.read_bytes()).hexdigest()
        if h != expected_sha:
            raise RuntimeError("T96_ARCHIVE_SHA_MISMATCH")

        saved = torch.load(
            path, map_location="cpu", weights_only=False
        )

        self.left = [
            tuple(row[:5]) for row in saved["LEFT"]
        ]
        self.right = [
            tuple(row[:5]) for row in saved["RIGHT"]
        ]

        if len(self.left) != 7000 or len(self.right) != 7000:
            raise RuntimeError("T96_ARCHIVE_COUNTS_INVALID")

    def mix(self, original, updates):
        n = len(original)

        if n < 32:
            raise RuntimeError("T96_BATCH_TOO_SMALL")

        each = int(round(0.30 * n))
        keep = n - 2 * each

        if keep < 1:
            raise RuntimeError("T96_MIX_RATIO_INVALID")

        rng = np.random.default_rng(
            202610289600000 + int(updates)
        )

        current_ids = rng.choice(
            n, size=keep, replace=False
        )
        left_ids = rng.choice(
            len(self.left), size=each, replace=False
        )
        right_ids = rng.choice(
            len(self.right), size=each, replace=False
        )

        mixed = (
            [original[int(i)] for i in current_ids]
            + [self.left[int(i)] for i in left_ids]
            + [self.right[int(i)] for i in right_ids]
        )

        order = rng.permutation(len(mixed))
        mixed = [mixed[int(i)] for i in order]

        if len(mixed) != n:
            raise RuntimeError("T96_BATCH_SIZE_CHANGED")

        return mixed, {
            "original": keep,
            "left": each,
            "right": each,
            "total": n,
        }
