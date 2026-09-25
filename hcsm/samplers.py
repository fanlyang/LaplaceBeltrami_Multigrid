"""The sampled error distribution, in coarse-space-complement form.

The specification is explicit that the model must not be trained on one
manufactured sine mode, and that the input to the DeepONet is the residual of
the *coarse-space-complement* part of an error, not of the error itself. This
module produces exactly that:

    e  ->  (e_C, e_F)  ->  r_F = A_epsilon e_F

The raw error family is the one the existing framework already uses -- smooth,
multiscale, localised, purely algebraic, and random mixtures of those four --
because it is documented, reproducible and deliberately not a single mode. What
changes here is what is done *after* the draw:

1. the error is canonicalised to unit A-energy,
2. it is split by the A-orthogonal decomposition of ``Problem.decompose``,
3. samples whose coarse-space-complement part is numerically degenerate are
   rejected and counted, and
4. the surviving e_F is canonicalised again, so every training target has
   ||e_F||_A = 1 and the Stage-I loss denominator is exactly one.

Step 3 matters and is not cosmetic. Projecting a *smooth* error onto the
complement of range(P) can leave almost nothing: that is the whole reason a
coarse-grid correction works. A sample whose e_F is 1e-14 of e carries no
information to learn from, and including it would let the loss be dominated by
division noise. The rejection rate per mechanism is reported, because it is
itself a measurement of how much of each error family the coarse space already
spans.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import deeponet_smoother as ds  # draw_sample, stable_seed, MECHANISMS

from .problem import Problem

#: Reject a draw when ||e_F||_A^2 / ||e||_A^2 falls below this. The error is
#: canonicalised first, so the denominator is one and this is an absolute
#: energy floor of 1e-8 on a unit-energy error, i.e. ||e_F||_A < 1e-4.
EF_REL_TOL = 1e-8

#: The default composition of every split. Kept identical across epsilons so a
#: difference between two epsilons cannot come from a different error mix.
DEFAULT_PLAN: Tuple[Tuple[str, int], ...] = (
    ("smooth", 64),
    ("multiscale", 128),
    ("localized", 96),
    ("algebraic", 128),
    ("mixed", 96),
)


@dataclass
class Split:
    """One role's samples, all in coarse-space-complement form."""

    E: np.ndarray          # the error itself, ||e||_A = 1
    E_C: np.ndarray        # its coarse-space component, ||e_C||_A = 1
    E_F: np.ndarray        # its coarse-space-complement part, ||e_F||_A = 1
    R_F: np.ndarray        # A_epsilon e_F
    mechanism: List[str]
    signature: List[str]
    params: List[str]
    ef_ratio: np.ndarray   # ||e_F||_A / ||e||_A for the accepted samples
    #: Content hash of each sample vector. This, not the signature, is what the
    #: disjointness audit gates on -- see ``split_audit``.
    digest: List[str] = field(default_factory=list)
    #: The exact RNG seed each sample's generator stream was built from.
    stream: List[int] = field(default_factory=list)
    rejected: Dict[str, int] = field(default_factory=dict)
    rejected_ratio: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )

    def __len__(self) -> int:
        return self.E.shape[0]

    def summary(self) -> Dict[str, object]:
        n_req = len(self) + sum(self.rejected.values())
        return {
            "n_samples": int(len(self)),
            "n_requested": int(n_req),
            "rejected_degenerate": int(sum(self.rejected.values())),
            "rejected_by_mechanism": dict(self.rejected),
            "ef_ratio_mean": float(self.ef_ratio.mean()) if len(self) else float("nan"),
            "ef_ratio_min": float(self.ef_ratio.min()) if len(self) else float("nan"),
            "ef_ratio_max": float(self.ef_ratio.max()) if len(self) else float("nan"),
        }


def draw_split(
    problem: Problem,
    plan: Sequence[Tuple[str, int]] = DEFAULT_PLAN,
    seed: int = 0,
    role: str = "train",
    ef_rel_tol: float = EF_REL_TOL,
    verbose: bool = False,
) -> Split:
    """Draw one role's samples.

    Each mechanism gets its own RNG, derived deterministically from
    (seed, role, mechanism, count), so the roles are independent streams and the
    whole thing is reproducible from the seed alone.
    """
    n = problem.n
    nyq = problem.nyquist
    angles = problem.angles

    E, E_C, E_F, R_F = [], [], [], []
    mech_list, sig_list, par_list, ratio_list = [], [], [], []
    digest_list: List[str] = []
    stream_list: List[int] = []
    rejected: Dict[str, int] = {}
    rejected_ratio: List[float] = []

    for mechanism, count in plan:
        stream_seed = ds.stable_seed(seed, role, mechanism, count)
        rng = np.random.default_rng(stream_seed)
        got, attempts = 0, 0
        while got < count and attempts < count * 40:
            attempts += 1
            vals, params, sig = ds.draw_sample(rng, mechanism, angles, nyq, n)

            e = problem.to_unit_energy(vals)
            if e is None:
                rejected[mechanism] = rejected.get(mechanism, 0) + 1
                continue

            e_C, e_F = problem.decompose(e)

            e_f_energy = problem.energy(e_F)
            if not np.isfinite(e_f_energy) or e_f_energy < ef_rel_tol:
                rejected[mechanism] = rejected.get(mechanism, 0) + 1
                rejected_ratio.append(
                    float(np.sqrt(max(e_f_energy, 0.0))) if np.isfinite(e_f_energy) else 0.0
                )
                continue

            e_F = e_F / np.sqrt(e_f_energy)
            e_C = problem.to_unit_energy(e_C)
            if e_C is None:                       # error is entirely in range(P)
                e_C = np.zeros(n)

            E.append(e)
            E_C.append(e_C)
            E_F.append(e_F)
            R_F.append(problem.A @ e_F)
            mech_list.append(mechanism)
            sig_list.append(repr(sig))
            par_list.append(json.dumps(params, default=float))
            ratio_list.append(float(np.sqrt(e_f_energy)))
            digest_list.append(_digest(e))
            stream_list.append(int(stream_seed))
            got += 1

        if got < count:
            raise SystemExit(
                "could not draw %d usable %s samples for %s (got %d after %d "
                "attempts); the coarse space spans almost all of this family"
                % (count, mechanism, role, got, attempts)
            )
        if verbose:
            print("      %-11s %4d accepted, %3d rejected" % (mechanism, got,
                                                              rejected.get(mechanism, 0)))

    return Split(
        E=np.asarray(E, dtype=np.float64),
        E_C=np.asarray(E_C, dtype=np.float64),
        E_F=np.asarray(E_F, dtype=np.float64),
        R_F=np.asarray(R_F, dtype=np.float64),
        mechanism=mech_list,
        signature=sig_list,
        params=par_list,
        ef_ratio=np.asarray(ratio_list, dtype=np.float64),
        digest=digest_list,
        stream=stream_list,
        rejected=rejected,
        rejected_ratio=np.asarray(rejected_ratio, dtype=np.float64),
    )


def controlled_modes(
    problem: Problem,
    wavenumbers: Optional[Sequence[int]] = None,
    family: str = "xi",
) -> Dict[str, object]:
    """The controlled toroidal test modes of the diagnostic section.

        e_k^(xi)(xi, eta) = sin(k xi),    e_k^(eta)(xi, eta) = sin(k eta)

    These are *diagnostic* errors only. They are never part of a training
    distribution -- the specification is explicit about that -- and each one is
    turned into its coarse-space-complement part by the same Galerkin projection
    as every other sample, then canonicalised to unit A-energy.

    ``family='xi'`` gives sin(k xi); ``family='eta'`` gives sin(k eta).
    Wavenumbers run to the lattice Nyquist limit, where sin(k xi) is still a
    genuine function of the torus but is no longer resolvable by the DoF lattice.
    """
    if wavenumbers is None:
        wavenumbers = tuple(range(1, problem.nyquist + 1))

    xi, eta = problem.angles[:, 0], problem.angles[:, 1]

    out = {"k": [], "E": [], "E_C": [], "E_F": [], "R_F": [], "ef_ratio": []}
    skipped = []
    for k in wavenumbers:
        vals = np.sin(k * xi) if family == "xi" else np.sin(k * eta)
        e = problem.to_unit_energy(vals)
        if e is None:
            skipped.append(int(k))
            continue
        e_C, e_F = problem.decompose(e)
        ef_energy = problem.energy(e_F)
        if not np.isfinite(ef_energy) or ef_energy < EF_REL_TOL:
            # sin(k .) lies (numerically) in range(P): the coarse space already
            # contains this mode exactly, so there is nothing to smooth.
            skipped.append(int(k))
            continue
        e_F = e_F / np.sqrt(ef_energy)
        e_C = problem.to_unit_energy(e_C)
        out["k"].append(int(k))
        out["E"].append(e)
        out["E_C"].append(e_C if e_C is not None else np.zeros(problem.n))
        out["E_F"].append(e_F)
        out["R_F"].append(problem.A @ e_F)
        out["ef_ratio"].append(float(np.sqrt(ef_energy)))

    out["skipped_k"] = skipped
    for key in ("E", "E_C", "E_F", "R_F"):
        out[key] = np.asarray(out[key], dtype=np.float64)
    for key in ("k", "ef_ratio"):
        out[key] = np.asarray(out[key], dtype=np.float64)
    return out


def _digest(v: np.ndarray) -> str:
    """A content hash of a sample vector: the identity of the function drawn."""
    h = hashlib.blake2b(np.ascontiguousarray(v, dtype=np.float64).tobytes(),
                        digest_size=16)
    return h.hexdigest()


def split_audit(splits: Dict[str, Split]) -> Dict[str, object]:
    """Prove that no sample is shared between two roles, two ways.

    THE GATE is the **content hash** of the sample vector: two roles must not
    contain the same function. If an RNG stream were accidentally reused, the
    draws would be bit-identical and this catches it exactly. (A full 1024-vector
    equality check, not a heuristic.)

    The signature count is reported as well, but it is **not** a gate, and the
    reason matters. A signature records the *generator parameters that are
    discrete* -- for the smooth family, the mode indices only. That family draws
    its modes from the small set {-2..2}^2 and then randomises amplitudes and
    phases, so two genuinely different functions routinely share a signature.
    Gating on it would abort runs that are perfectly sound; it is kept as a
    diagnostic, not as the check. The first version of this function did gate on
    it and did exactly that.
    """
    owner: Dict[str, str] = {}
    duplicates: List[str] = []
    for role, s in splits.items():
        for d in s.digest:
            if owner.setdefault(d, role) != role:
                duplicates.append(d)

    sig_owner: Dict[str, str] = {}
    sig_overlaps = 0
    for role, s in splits.items():
        for sig in s.signature:
            if sig_owner.setdefault(sig, role) != role:
                sig_overlaps += 1

    # The structural guarantee behind the hash check: every (role, mechanism)
    # pair gets its own RNG stream, so the roles cannot share draws.
    streams = {role: sorted(set(s.stream)) for role, s in splits.items()}
    flat = [x for v in streams.values() for x in v]
    stream_collisions = len(flat) - len(set(flat))

    return {
        "cross_split_exact_duplicates": len(duplicates),
        "cross_split_signature_overlaps": sig_overlaps,
        "signature_overlap_note": "diagnostic only, not a gate: the smooth "
                                  "family's signature records modes but not "
                                  "amplitudes or phases",
        "role_stream_seeds": streams,
        "role_stream_seed_collisions": stream_collisions,
        "unique_signatures": len(sig_owner),
        "total_samples": sum(len(s) for s in splits.values()),
    }
