"""Focused regressions: python -m unittest hcsm.test_corrected_pilot -v."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch

from .train import TrainConfig, train_penalised, train_smoother, validation_means
from .run_tensor import Mixture
from .aniso_errors import draw_mixture, equal_plan
from .aniso_evaluate import jacobi_bound_check, evaluate_families
from .evaluate import MethodSet, Smoothers, _corrections
from .problem import load_problem
from .losses import QProjector, ScaleEquivariant
from .model import build_model, build_features
from .vcycle import Hierarchy, ClassicalSmoothing, solve_with_preconditioner


class LinearModel(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.layer = torch.nn.Linear(n, n, bias=False).double()

    def forward(self, r, features, coeff=None):
        return self.layer(r)


class CorrectedPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.problem = load_problem("level_data", 1e-4, level=1, verbose=False)

    def test_validation_includes_all_family_blocks_and_partial_batch(self):
        r = torch.zeros(519, 2, dtype=torch.float64)
        e = torch.cat([torch.full((128, 2), float(i), dtype=torch.float64)
                       for i in range(4)] + [torch.full((7, 2), 9., dtype=torch.float64)])
        got, = validation_means(LinearModel(2), r, e, None, 128,
                               lambda c, target: ((target-c)**2).mean())
        self.assertAlmostEqual(got, float((e**2).mean()), places=12)

    def test_patience_units_and_best_checkpoint_restoration(self):
        p = self.problem
        fam = draw_mixture(p, equal_plan(8), 33, "regression")
        data = Mixture(fam)
        model = LinearModel(p.n)
        cfg = TrainConfig(epochs=10, log_every=1, early_stop=3,
                          lr_patience_checks=1, batch_size=4)
        with tempfile.TemporaryDirectory() as out:
            # Force validation deterioration: first checkpoint must be restored.
            with patch("hcsm.train.validation_means", side_effect=[(v,v,v) for v in (1.,2.,3.,4.)]):
                history, info = train_penalised(model, p, data, data, cfg,
                    QProjector(p), 1., out_dir=out, log=lambda _: None)
            saved = torch.load(Path(out)/"best_model.pt", weights_only=True)
            self.assertEqual(info["best_epoch"], 1)
            self.assertEqual(info["optimizer_steps"], 4)
            self.assertLess(history[-1]["lr"], cfg.lr)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, saved["state_dict"][name], rtol=0, atol=0)
            self.assertFalse(model.training)

    def test_exact_optimizer_update_budget(self):
        p = self.problem
        data = Mixture(draw_mixture(p, equal_plan(8), 19, "budget"))
        with tempfile.TemporaryDirectory() as out:
            _, info = train_penalised(LinearModel(p.n), p, data, data,
                TrainConfig(epochs=3, log_every=2), QProjector(p), 1.,
                out_dir=out, log=lambda _: None)
        self.assertEqual(info["optimizer_steps"], 3)

    def test_scalar_trainer_checkpoint_uses_safe_serializable_metadata(self):
        p = self.problem
        mix = Mixture(draw_mixture(p, equal_plan(8), 31, "scalar-trainer"))
        data = SimpleNamespace(E_F=mix.E, R_F=mix.R)
        with tempfile.TemporaryDirectory() as out:
            model = LinearModel(p.n)
            _, info = train_smoother(model, p, data, data,
                TrainConfig(epochs=2, log_every=1), out_dir=out, log=lambda _: None)
            saved = torch.load(info["checkpoint"], weights_only=True)
            self.assertIsInstance(saved["val_loss"], float)
            self.assertFalse(model.training)

    def test_projected_bound_and_invalid_inputs(self):
        p = self.problem
        methods = MethodSet(Smoothers(p.A), {}, 2/3, 1.)
        e = Mixture(draw_mixture(p, equal_plan(8), 71, "bound")).E
        got = jacobi_bound_check(p, methods, e)
        self.assertEqual(got["fraction_where_bound_holds"], 1.)
        with self.assertRaises(ValueError):
            jacobi_bound_check(p, methods, np.ones((1,p.n)))

    def test_measured_corrections_unchanged_by_timing(self):
        p = self.problem
        families = draw_mixture(p, equal_plan(8), 93, "timing")
        methods = MethodSet(Smoothers(p.A), {}, 2/3, 1.)
        measured = evaluate_families(p, methods, families)
        for name, family in families.items():
            for method, correction in _corrections(p, methods, family.E).items():
                expected = np.mean(p.norm_A(family.E-correction))
                self.assertAlmostEqual(measured[name][method]["full"].mean, expected)
                self.assertGreater(measured[name][method]["seconds"], 0.)

    def test_fgmres_handles_nonlinear_preconditioner_and_true_residual(self):
        a = sp.diags(np.linspace(1.,8.,8), format="csr")
        h = Hierarchy([a], [None])
        def nonlinear_cycle(hierarchy, smoother, level, r):
            return r / a.diagonal() * (1. + .2*np.tanh(r))
        with patch("hcsm.vcycle.vcycle", side_effect=nonlinear_cycle):
            result = solve_with_preconditioner(h, object(), 0, np.ones(8),
                                              flexible=True, maxiter=8, tol=1e-10)
        self.assertTrue(result["converged"], result)
        self.assertLessEqual(result["residual"], 1e-10)
        self.assertEqual(result["solver"], "fgmres")

    def test_forward_gs_cannot_enter_pcg(self):
        a = sp.diags([1.,2.,3.],format="csr")
        h = Hierarchy([a],[None])
        result = solve_with_preconditioner(h, ClassicalSmoothing(h,"gs"),0,
                                          np.ones(3), flexible=False, maxiter=3)
        self.assertEqual(result["solver"], "fgmres")
        self.assertTrue(result["converged"])

    def test_residual_interface_and_scaling(self):
        from .run_tensor import WrappedSmoother
        p = self.problem
        features = build_features(p)
        model = ScaleEquivariant(build_model(p, features.shape[1], base_smoother="jacobi"))
        wrapped = WrappedSmoother(model, features)
        e = np.random.default_rng(21).normal(size=p.n)
        r = p.A @ e
        methods = MethodSet(Smoothers(p.A), {"test": wrapped}, 2/3, 1.)
        got = _corrections(p, methods, e[None,:])["deeponet_test"][0]
        np.testing.assert_allclose(got, wrapped.from_residual(r))
        np.testing.assert_allclose(wrapped.from_residual(1e-5*r),1e-5*got,rtol=1e-10,atol=1e-15)


if __name__ == "__main__":
    unittest.main()
