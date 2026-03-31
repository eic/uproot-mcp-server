"""Create a synthetic ROOT file for use in unit tests.

Run this script once to regenerate the test fixture::

    python create_fixture.py

The file ``test_eic.root`` contains a TTree named ``events`` with 1000 entries
and five branches: px, py, pz (float32 momentum components), charge (int32),
and energy (float32, derived quantity).
"""

from __future__ import annotations

import pathlib

import numpy as np
import uproot

FIXTURE_PATH = pathlib.Path(__file__).parent / "test_eic.root"
N_EVENTS = 1000


def create_fixture(path: pathlib.Path = FIXTURE_PATH) -> None:
    rng = np.random.default_rng(42)
    n = N_EVENTS

    px = rng.normal(0, 1, n).astype(np.float32)
    py = rng.normal(0, 1, n).astype(np.float32)
    pz = rng.normal(5, 2, n).astype(np.float32)
    charge = rng.choice([-1, 0, 1], size=n).astype(np.int32)
    energy = np.sqrt(px**2 + py**2 + pz**2 + 0.139**2).astype(np.float32)

    with uproot.recreate(str(path)) as f:
        f.mktree(
            "events",
            {
                "px": "float32",
                "py": "float32",
                "pz": "float32",
                "charge": "int32",
                "energy": "float32",
            },
            title="Simulated EIC events",
        )
        f["events"].extend(
            {"px": px, "py": py, "pz": pz, "charge": charge, "energy": energy}
        )


if __name__ == "__main__":
    create_fixture()
    print(f"Created {FIXTURE_PATH}")
