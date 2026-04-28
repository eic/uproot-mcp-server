"""Create synthetic ROOT files for use in unit tests.

Run this script once to regenerate the test fixtures::

    python create_fixture.py

Two files are produced:

- ``test_eic.root`` — TTree ``events`` with 1000 entries and five flat
  branches: px, py, pz (float32), charge (int32), energy (float32).
- ``test_eic_nested.root`` — TTree ``events`` with a nested record branch
  ``hits`` whose sub-fields ``x``/``y`` appear only under recursive listing.
  The full-path keys of ``hits`` use ``.`` as a separator (e.g. ``hits.x``),
  but the leaf-only name ``x`` is in ``tree.keys(recursive=True,
  full_paths=False)`` while ``set(tree.keys())`` does not contain it — the
  same name-resolution mismatch that bites podio/edm4eic files where the full
  path uses ``/`` (``Particles/Particles.momentum.x``) yet ``tree[
  "Particles.momentum.x"]`` resolves.
"""

from __future__ import annotations

import pathlib

import awkward as ak
import numpy as np
import uproot

FIXTURE_PATH = pathlib.Path(__file__).parent / "test_eic.root"
NESTED_FIXTURE_PATH = pathlib.Path(__file__).parent / "test_eic_nested.root"
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


def create_nested_fixture(path: pathlib.Path = NESTED_FIXTURE_PATH) -> None:
    rng = np.random.default_rng(43)
    n = N_EVENTS

    hits = ak.zip(
        {
            "x": rng.normal(0, 1, n).astype(np.float32),
            "y": rng.normal(0, 1, n).astype(np.float32),
        }
    )
    flat = np.arange(n, dtype=np.float32)

    with uproot.recreate(str(path)) as f:
        f["events"] = {"hits": hits, "flat": flat}


if __name__ == "__main__":
    create_fixture()
    print(f"Created {FIXTURE_PATH}")
    create_nested_fixture()
    print(f"Created {NESTED_FIXTURE_PATH}")
