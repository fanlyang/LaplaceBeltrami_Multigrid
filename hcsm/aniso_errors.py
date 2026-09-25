"""The four training-error families, and the held-out stress set.

Every candidate ``z`` is pushed through the complement projector and rescaled
before it is used, exactly as the specification says:

    e_tilde = Q_epsilon z          (by coarse solve, no inverse formed)
    e       = e_tilde / ||e_tilde||_{A_epsilon}
    r       = A_epsilon e

A candidate whose projection is numerically negligible is DISCARDED and counted,
because such a sample carries no information and would only add division noise.
The ratio ||Q z||_A / ||z||_A is recorded for every candidate -- accepted or
discarded -- so the discard threshold is visible in the results rather than
buried in the code.

The four families
-----------------
``gaussian``   independent Gaussian nodal values. The classical algebraic error:
               rough in both directions, no structure to exploit.
``fourier``    random mixtures of sin/cos modes in both angles, with random
               phases and decaying amplitudes, spanning the mesh-resolved band.
``weak``       oscillatory in xi, constant or slowly varying in eta -- i.e.
               oscillatory along the WEAK direction of the tensor and smooth
               along the strong one. This is the family the standard
               anisotropic-diffusion analysis says a point smoother handles
               worst, because such an error is nearly invisible to the operator
               while being unreachable by the coarse grid.
``survivor``   what a classical smoother leaves behind: 2, 5 or 10 steps of
               damped Jacobi, forward Gauss-Seidel or symmetric Gauss-Seidel
               applied to a projected random error. These are, by construction,
               the directions a classical method does NOT reduce.

**The ``weak`` family is not assumed to be difficult.** It is generated and then
measured like every other family; the per-family reduction table is what decides
whether it is. Two things are separately reported so that this cannot be
confused: how hard the family is for the classical smoothers, and how much
training coverage it got.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .problem import Problem

#: Families and their equal proportions in the training mixture.
FAMILIES: Tuple[str, ...] = ("gaussian", "fourier", "weak", "survivor")

#: Sweep counts used by the survivor family, and by the classical baselines.
SURVIVOR_SWEEPS: Tuple[int, ...] = (2, 5, 10)
SURVIVOR_METHODS: Tuple[str, ...] = ("jacobi", "gs", "sgs")

#: Reject a candidate whose projection is this small: ||Q z||_A / ||z||_A below
#: this means z was almost entirely inside range(P), so its complement part
#: carries no information to learn from.
PROJECTION_TOL = 1e-6


@dataclass
class Family:
    """One family's projected, normalised samples."""

    name: str
    E: np.ndarray            # e, ||e||_A = 1
    R: np.ndarray            # A_epsilon e
    Qz_ratio: np.ndarray     # ||Q z||_A / ||z||_A, accepted samples
    subkind: List[str]       # e.g. 'jacobi-5' for a survivor
    digest: List[str]
    n_discarded: int = 0
    discarded_ratio: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64))

    def __len__(self) -> int:
        return self.E.shape[0]


def _digest(v: np.ndarray) -> str:
    return hashlib.blake2b(np.ascontiguousarray(v, dtype=np.float64).tobytes(),
                           digest_size=16).hexdigest()


# ---------------------------------------------------------------------------
# Generators, before projection
# ---------------------------------------------------------------------------


def gen_gaussian(rng: np.random.Generator, problem: Problem) -> Tuple[np.ndarray, str, dict]:
    return rng.normal(size=problem.n), "gaussian", {}


def gen_fourier(rng: np.random.Generator, problem: Problem,
                n_modes: int = 6) -> Tuple[np.ndarray, str, dict]:
    """Random Fourier mixture in both angles.

    Wavenumbers are drawn from the MESH-RESOLVED band only, |m| <= nyquist and
    |n| <= nyquist. Sampling a mode above the lattice Nyquist aliases it: on an
    n_side lattice cos(k xi) and cos((n_side - k) xi) take identical values at
    every DoF, so a generator that claims to have drawn k > nyquist would have
    silently produced a different, smoother function and recorded a false
    provenance. ``assert_no_aliasing`` checks this rather than trusting it.
    """
    nyq = problem.nyquist
    xi, eta = problem.angles[:, 0], problem.angles[:, 1]

    modes, amps, phases = [], [], []
    for _ in range(int(rng.integers(3, n_modes + 1))):
        m = int(rng.integers(-nyq, nyq + 1))
        n = int(rng.integers(-nyq, nyq + 1))
        if m == 0 and n == 0:
            continue
        modes.append((m, n))
        # Decaying amplitude: a broad but not flat frequency distribution, so
        # the mixture is not dominated by the highest modes.
        amps.append(float(np.hypot(m, n) ** (-0.8)) * float(rng.normal()))
        phases.append(float(rng.uniform(0.0, 2 * np.pi)))
    if not modes:
        modes, amps, phases = [(1, 1)], [1.0], [0.0]

    field = np.zeros_like(xi)
    for (m, n), a, ph in zip(modes, amps, phases):
        field += a * np.cos(m * xi + n * eta + ph)

    return field, "fourier", {"modes": modes,
                              "amps": [float(a) for a in amps],
                              "phases": [float(p) for p in phases],
                              "nyquist": int(nyq)}


def gen_weak(rng: np.random.Generator, problem: Problem) -> Tuple[np.ndarray, str, dict]:
    """Oscillatory in xi, constant or slowly varying in eta.

    The xi wavenumber is drawn from the upper half of the resolved band (so it
    is genuinely oscillatory in the weak direction), and the eta content is
    either nothing at all or a low mode. ``eta_mode='constant'`` is the extreme
    case: the error is exactly constant along the strong direction.
    """
    nyq = problem.nyquist
    xi, eta = problem.angles[:, 0], problem.angles[:, 1]

    m = int(rng.integers(max(2, nyq // 2), nyq + 1))
    phase = float(rng.uniform(0.0, 2 * np.pi))
    field = np.cos(m * xi + phase)

    if rng.random() < 0.5:
        kind = "constant-in-eta"
        n = 0
    else:
        kind = "slow-in-eta"
        n = int(rng.integers(1, max(2, nyq // 4) + 1))
        field = field * np.cos(n * eta + float(rng.uniform(0.0, 2 * np.pi)))

    return field, "weak-" + kind, {"m": m, "n": n, "phase": phase,
                                   "nyquist": int(nyq)}


def gen_random_base(rng: np.random.Generator, problem: Problem) -> np.ndarray:
    """A projected random error, used as the seed for the survivor family."""
    return rng.normal(size=problem.n)


# ---------------------------------------------------------------------------
# Aliasing check
# ---------------------------------------------------------------------------


def assert_no_aliasing(problem: Problem, modes: Sequence[Tuple[int, int]]) -> None:
    """Refuse a Fourier mode the DoF lattice cannot represent distinctly.

    On an ``n_side`` lattice the mode ``cos(k xi)`` takes exactly the same
    values at every DoF as ``cos((n_side - k) xi)``. So a generator that draws
    ``k > nyquist = n_side/2`` produces a function that is NOT the one its
    recorded parameters describe, and any conclusion drawn from that sample is
    about an unknown function. The generator above caps at nyquist; this asserts
    it, so the cap cannot be quietly broken later.
    """
    nyq = problem.nyquist
    for (m, n) in modes:
        for k in (m, n):
            if abs(k) > nyq:
                raise SystemExit(
                    "Fourier mode %d exceeds the lattice Nyquist %d: the sample "
                    "would alias to a different frequency than the one "
                    "recorded" % (k, nyq))


# ---------------------------------------------------------------------------
# The survivor family
# ---------------------------------------------------------------------------


def _survive(problem: Problem, z: np.ndarray, method: str, sweeps: int,
             omega: float) -> np.ndarray:
    """Apply a classical smoother to z, ``sweeps`` times, and return the survivor.

    Operates on the projected error, which is what a smoother inside a V-cycle
    actually sees after the coarse grid has done its part.
    """
    from .evaluate import Smoothers

    sm = Smoothers(problem.A)
    cur = z.copy()
    for _ in range(sweeps):
        r = problem.A @ cur
        if method == "jacobi":
            cur = cur - sm.jacobi(r, omega)
        elif method == "gs":
            cur = cur - sm.gs(r)
        elif method == "sgs":
            cur = cur - sm.sgs(r)
        else:
            raise ValueError(method)
    return cur


# ---------------------------------------------------------------------------
# Drawing one family
# ---------------------------------------------------------------------------


def draw_family(
    problem: Problem,
    family: str,
    count: int,
    seed: int,
    role: str,
    jacobi_omega: float = 2.0 / 3.0,
    verbose: bool = False,
) -> Family:
    """Draw ``count`` accepted samples of one family, projected and normalised.

    Each (role, family) pair gets its own RNG stream, so the roles are
    independent by construction; the content hash of every accepted sample is
    recorded so the caller can prove no sample appears in two roles.
    """
    stream_seed = int(hashlib.blake2b(
        ("%d|%s|%s|%d" % (seed, role, family, count)).encode(),
        digest_size=4).hexdigest(), 16)
    rng = np.random.default_rng(stream_seed)

    E, R, ratios, kinds, digests = [], [], [], [], []
    discarded = 0
    discarded_ratios: List[float] = []
    got, attempts = 0, 0

    while got < count and attempts < count * 60:
        attempts += 1

        if family == "survivor":
            method = SURVIVOR_METHODS[int(rng.integers(0, len(SURVIVOR_METHODS)))]
            sweeps = SURVIVOR_SWEEPS[int(rng.integers(0, len(SURVIVOR_SWEEPS)))]
            base = gen_random_base(rng, problem)
            # The survivor is taken from a projected error, so the smoother is
            # acting where a smoother actually acts.
            base_proj = problem.apply_Q(base)
            nb = np.linalg.norm(problem.A @ base_proj)
            if nb > 0:
                base_proj = base_proj / np.sqrt(
                    max(problem.energy(base_proj), 1e-300))
            z = _survive(problem, base_proj, method, sweeps, jacobi_omega)
            kind = "%s-%d" % (method, sweeps)
            params = {"method": method, "sweeps": sweeps, "omega": jacobi_omega}
        elif family == "gaussian":
            z, kind, params = gen_gaussian(rng, problem)
        elif family == "fourier":
            z, kind, params = gen_fourier(rng, problem)
            assert_no_aliasing(problem, params["modes"])
        elif family == "weak":
            z, kind, params = gen_weak(rng, problem)
        else:
            raise ValueError("unknown family %r" % family)

        nz = problem.norm_A(z)
        if not np.isfinite(nz) or nz <= 0.0:
            discarded += 1
            continue

        projection = problem.apply_Q(z)
        ratio = problem.norm_A(projection) / nz

        if not np.isfinite(ratio) or ratio < PROJECTION_TOL:
            discarded += 1
            discarded_ratios.append(float(ratio) if np.isfinite(ratio) else 0.0)
            continue

        e = projection / np.sqrt(max(problem.energy(projection), 1e-300))
        E.append(e)
        R.append(problem.A @ e)
        ratios.append(float(ratio))
        kinds.append(kind)
        digests.append(_digest(e))
        got += 1

    if got < count:
        raise SystemExit(
            "family %r produced only %d of %d usable %s samples after %d "
            "attempts; raise the attempt cap or lower PROJECTION_TOL"
            % (family, got, count, role, attempts))

    if verbose:
        print("        %-9s %4d accepted, %3d discarded (median ||Qz||/||z|| "
              "%.3f)" % (family, got, discarded,
                         float(np.median(ratios)) if ratios else float("nan")))

    return Family(name=family,
                  E=np.asarray(E, dtype=np.float64),
                  R=np.asarray(R, dtype=np.float64),
                  Qz_ratio=np.asarray(ratios, dtype=np.float64),
                  subkind=kinds,
                  digest=digests,
                  n_discarded=discarded,
                  discarded_ratio=np.asarray(discarded_ratios, dtype=np.float64))


def draw_mixture(
    problem: Problem,
    plan: Dict[str, int],
    seed: int,
    role: str,
    jacobi_omega: float = 2.0 / 3.0,
    verbose: bool = False,
) -> Dict[str, Family]:
    """Draw every family in ``plan``, each with its own RNG stream."""
    return {
        family: draw_family(problem, family, count, seed, role,
                            jacobi_omega=jacobi_omega, verbose=verbose)
        for family, count in plan.items()
    }


def equal_plan(total: int) -> Dict[str, int]:
    """Equal proportions across the four families.

    The specification asks for equal proportions to start with. The counts are
    equal here; what is NOT equal is how much of each family survives the
    projection, and that is reported per family rather than silently rebalanced
    -- rebalancing by survival would bias the mixture toward whichever family
    happens to be least represented in range(P).
    """
    per = max(1, total // len(FAMILIES))
    return {f: per for f in FAMILIES}


def audit(families: Dict[str, Family]) -> Dict[str, object]:
    """Cross-role duplicate check on the sample content hashes."""
    seen: Dict[str, str] = {}
    duplicates = 0
    for role, fam in families.items():
        for d in fam.digest:
            if seen.setdefault(d, role) != role:
                duplicates += 1
    return {
        "cross_role_exact_duplicates": duplicates,
        "total_samples": sum(len(f) for f in families.values()),
        "by_family": {name: {"n": len(f), "discarded": f.n_discarded,
                             "median_Qz_ratio": float(np.median(f.Qz_ratio))
                             if len(f) else float("nan")}
                      for name, f in families.items()},
    }


# ---------------------------------------------------------------------------
# The held-out stress set
# ---------------------------------------------------------------------------

#: The wavenumbers of the stress set. Fixed in advance, not chosen by looking at
#: the test results.
STRESS_XI_MODES: Tuple[int, ...] = (1, 2, 4, 8, 12, 16)
STRESS_ETA_MODES: Tuple[int, ...] = (0, 1, 2, 4)


def stress_set(problem: Problem) -> Dict[str, object]:
    """Independently computed near-worst-case classical modes, held out.

    These are the analytic modes the anisotropic-diffusion analysis predicts a
    point smoother handles worst: oscillatory in xi (the weak direction) and
    constant or slowly varying in eta (the strong direction). They are generated
    from a closed form, never sampled from the training distribution, and they
    are used for NOTHING except the final stress evaluation -- no model
    selection, no threshold, no hyperparameter sees them.

    Each mode is projected through the same Q_epsilon and normalised in A, so it
    is a genuine admissible error and not a raw analytic mode.
    """
    xi, eta = problem.angles[:, 0], problem.angles[:, 1]
    E, R, labels, ratios = [], [], [], []

    skipped = []
    for m in STRESS_XI_MODES:
        for n in STRESS_ETA_MODES:
            if m == 0 and n == 0:
                continue
            z = np.cos(m * xi) * (np.cos(n * eta) if n else 1.0)
            nz = problem.norm_A(z)
            if not np.isfinite(nz) or nz <= 0:
                skipped.append((m, n))
                continue
            proj = problem.apply_Q(z)
            ratio = problem.norm_A(proj) / nz
            if not np.isfinite(ratio) or ratio < PROJECTION_TOL:
                skipped.append((m, n))
                continue
            e = proj / np.sqrt(max(problem.energy(proj), 1e-300))
            E.append(e)
            R.append(problem.A @ e)
            labels.append("cos(%d xi)cos(%d eta)" % (m, n))
            ratios.append(float(ratio))

    return {
        "E": np.asarray(E, dtype=np.float64),
        "R": np.asarray(R, dtype=np.float64),
        "label": labels,
        "mode_xi": [int(l.split("cos(")[1].split(" ")[0]) for l in labels],
        "mode_eta": [int(l.split("cos(")[2].split(" ")[0]) for l in labels],
        "Qz_ratio": np.asarray(ratios, dtype=np.float64),
        "skipped": skipped,
    }
