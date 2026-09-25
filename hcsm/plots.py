"""Every figure the specification asks for.

Design decisions that apply to all of them:

* One palette, one series order, one meaning per colour, fixed across figures
  (``style.py``). A colour never changes meaning between panels.
* The categorical set is capped at four slots because that is the largest set
  that clears the all-pairs colour-vision-deficiency floors against this surface.
  A fifth method is faceted, not recoloured.
* Every curve is also directly labelled, and every plotted number also exists in
  a CSV under ``results/``. That is the relief the contrast rule requires for the
  lighter slots, and it is what makes a figure readable in greyscale.
* Error fields are *signed*, so they use the diverging map with its neutral grey
  at zero. Coefficient magnitude uses the single-hue sequential ramp. Neither is
  a rainbow.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from . import style
from .evaluate import MethodSet
from .problem import Problem, lattice_grid, torus_surface

style.apply()

#: The four methods drawn on the main comparison figures, in fixed order.
#:
#: The DeepONet column is `deeponet_skip` -- the framework's own best-measured
#: configuration -- and that choice is fixed here, in advance. It is NOT the
#: variant that happened to score best on the test set, which would be selecting
#: on the test set. The other two variants are reported in every table and in
#: their own figure (``plot_variant_bars``).
CURVE_METHODS: Tuple[str, ...] = ("jacobi", "gs", "sgs_ssor", "deeponet_skip")

#: The three DeepONet variants against Jacobi, for the ablation figure. Four
#: series, so it stays inside the validated all-pairs slot cap: one hue for
#: Jacobi and one for the DeepONet family, with the dash carrying the variant.
VARIANT_METHODS: Tuple[str, ...] = ("jacobi", "deeponet_plain", "deeponet_skip",
                                    "deeponet_skip_fourier")


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print("    wrote %s" % path)


# ---------------------------------------------------------------------------
# 1. kappa_epsilon on the torus
# ---------------------------------------------------------------------------


def plot_kappa_torus(problems: Sequence[Problem], path: str) -> None:
    """kappa_epsilon on the torus, one panel per epsilon.

    Row 1 draws the coefficient on the actual torus surface; row 2 draws the same
    field in the (xi, eta) parameter rectangle, where it is a smooth function of
    two independent period-2*pi angles and the shape is easier to read. Each panel
    carries its own colour scale, because the point of the figure is precisely
    that the *pattern* is identical while the *range* is not: kappa_epsilon is
    kappa_1e-1 shifted down by a constant, so the contrast (1+eps)/eps is the only
    thing that changes.
    """
    n = len(problems)
    fig = plt.figure(figsize=(2.75 * n, 5.4))
    cmap = style.sequential_cmap()
    X, Y, Z = torus_surface(problems[0])

    for j, p in enumerate(problems):
        _, _, V = lattice_grid(p, p.coeff)
        # Each panel gets its OWN scale. That is the point of the figure:
        # kappa_epsilon is kappa_1e-1 shifted down by a constant, so the pattern
        # is identical and only the range -- hence the contrast -- changes.
        norm = plt.Normalize(vmin=V.min(), vmax=V.max())

        ax = fig.add_subplot(2, n, j + 1, projection="3d")
        ax.plot_surface(X, Y, Z, facecolors=cmap(norm(V)), rstride=1, cstride=1,
                        shade=False, antialiased=False, linewidth=0)
        ax.set_axis_off()
        ax.set_box_aspect(TORUS_ASPECT, zoom=TORUS_ZOOM)
        ax.view_init(**TORUS_VIEW)
        ax.set_title(r"$\varepsilon = 10^{%d}$,  contrast %.0f$\times$"
                     % (int(round(np.log10(p.eps))), (1.0 + p.eps) / p.eps),
                     fontsize=8.5, color=style.INK, pad=2)

        ax2 = fig.add_subplot(2, n, n + j + 1)
        im = ax2.imshow(V, origin="lower", cmap=cmap, norm=norm,
                        extent=[-np.pi, np.pi, -np.pi, np.pi], aspect="equal",
                        interpolation="nearest")
        ax2.set_xlabel(r"$\xi$")
        if j == 0:
            ax2.set_ylabel(r"$\eta$")
        ax2.set_xticks([-np.pi, 0, np.pi])
        ax2.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
        ax2.set_yticks([-np.pi, 0, np.pi])
        ax2.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
        ax2.grid(False)
        style.style_axes(ax2)
        cb = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7, length=2)
        cb.outline.set_visible(False)

    fig.suptitle(r"$\kappa_\varepsilon(\xi,\eta) = \varepsilon + \sin^2\xi\,\cos^2\eta$"
                 "   (identical pattern, four different contrasts; each panel is "
                 "scaled to its own range)",
                 fontsize=10, fontweight="semibold", color=style.INK, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, path)


# ---------------------------------------------------------------------------
# 2-4. epsilon vs the reduction statistics
# ---------------------------------------------------------------------------


def plot_eps_curves(
    summary: Dict[str, Dict[str, Dict[str, float]]],
    section: str,
    metric: str,
    label: str,
    path: str,
    title: str,
    methods: Sequence[str] = CURVE_METHODS,
    reference_line: Optional[float] = None,
    reference_label: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """One reduction statistic against epsilon, log-scaled in epsilon.

    ``section`` names which results block to read -- ``rho_F``, ``rho_C`` or
    ``rho_TG`` -- and is a parameter rather than something inferred from the
    title. It was inferred from the title in an earlier version, which meant a
    reworded caption could silently switch a figure to the wrong data.
    ``summary[eps_tag][section][method]`` -> statistic dict.

    The epsilon axis is logarithmic because the contrast (1+eps)/eps is what the
    experiment varies, and that is a ratio.
    """
    eps_tags = list(summary.keys())
    eps_values = np.array([float(t) for t in eps_tags])

    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    if reference_line is not None:
        ax.axhline(reference_line, color=style.INK_MUTED, lw=1.2, ls="--", zorder=1)
        ax.annotate(reference_label or "", xy=(eps_values[0], reference_line),
                    xytext=(2, 4), textcoords="offset points",
                    fontsize=7.5, color=style.INK_MUTED)

    for mi, meth in enumerate(methods):
        ys = []
        for t in eps_tags:
            cell = summary[t].get(section, {}).get(meth)
            ys.append(np.nan if cell is None else cell[metric])
        ys = np.array(ys, dtype=np.float64)
        st = style.METHOD_STYLE.get(meth, {})
        ax.plot(eps_values, ys, color=style.METHOD_COLOR.get(meth, style.CATEGORICAL[mi % 4]),
                ls=st.get("ls", "-"), marker=st.get("marker", "o"),
                markeredgecolor=style.SURFACE, markeredgewidth=1.2, zorder=3,
                label=style.METHOD_LABEL.get(meth, meth))
        style.label_line(ax, eps_values[-1], ys[-1],
                         style.METHOD_SHORT.get(meth, meth),
                         style.METHOD_COLOR.get(meth, style.CATEGORICAL[mi % 4]))

    ax.set_xscale("log")
    ax.set_xticks(eps_values)
    ax.set_xticklabels([r"$10^{%d}$" % int(round(np.log10(v))) for v in eps_values])
    ax.set_xlabel(r"coefficient offset $\varepsilon$   (contrast $(1+\varepsilon)/\varepsilon$ grows to the left)")
    ax.set_ylabel(label)
    ax.set_title(title, loc="left")
    if ylim is not None:
        ax.set_ylim(*ylim)
    style.style_axes(ax)
    ax.legend(loc="upper right", ncols=2)
    ax.margins(x=0.16)
    fig.tight_layout()
    _save(fig, path)



# ---------------------------------------------------------------------------
# 5-6. controlled modes
# ---------------------------------------------------------------------------


def plot_mode_scan(
    scans: Dict[float, Dict[str, Dict[str, object]]],
    family: str,
    path: str,
    methods: Sequence[str] = CURVE_METHODS,
) -> None:
    """k vs rho_k for sin(k xi) or sin(k eta), one panel per epsilon.

    A flat curve at 1 means the smoother does not touch that mode at all; a curve
    above 1 means it amplifies it. The vertical position of the whole curve is
    what a two-grid method would feel.
    """
    eps_tags = list(scans.keys())
    n = len(eps_tags)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.7), sharey=True)
    axes = np.atleast_1d(axes)      # a single epsilon gives one bare Axes

    for j, tag in enumerate(eps_tags):
        ax = axes[j]
        rec = scans[tag].get(family, {})
        k = np.asarray(rec.get("k", []), dtype=np.float64)
        if k.size == 0:
            ax.set_axis_off()
            continue
        ax.axhline(1.0, color=style.INK_MUTED, lw=1.0, ls="--", zorder=1)
        for mi, meth in enumerate(methods):
            y = np.asarray(rec.get(meth, np.full(k.size, np.nan)), dtype=np.float64)
            st = style.METHOD_STYLE.get(meth, {})
            ax.plot(k, y, color=style.METHOD_COLOR.get(meth), ls=st.get("ls", "-"),
                    marker=st.get("marker", "o"), markeredgecolor=style.SURFACE,
                    markeredgewidth=1.0, label=style.METHOD_LABEL.get(meth, meth),
                    zorder=3)
        ax.set_title(r"$\varepsilon = 10^{%d}$" % int(round(np.log10(float(tag)))))
        ax.set_xlabel(r"wavenumber $k$")
        if j == 0:
            ax.set_ylabel(r"$\rho_k$")
        style.style_axes(ax)
        ax.set_xticks(list(k))

    axes[0].legend(loc="upper left", fontsize=7)
    modes = r"\sin(k\xi)" if family == "xi" else r"\sin(k\eta)"
    fig.suptitle("Controlled mode diagnostic: coarse-space-complement part of "
                 "$%s$,  $\\rho_k = \\|S(e_{k,F})\\|_A / \\|e_{k,F}\\|_A$" % modes,
                 fontsize=10, fontweight="semibold", color=style.INK, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    _save(fig, path)


# ---------------------------------------------------------------------------
# 7. the Jacobi-hard errors, on the torus
# ---------------------------------------------------------------------------


#: Camera for the torus panels. The ring lies in the x-z plane, so looking along
#: -y (azim = -90) shows the hole; the small elevation lifts it off face-on so the
#: tube is visible too. The box aspect matches the data ranges x,z in [-3,3] and
#: y in [-1,1]; forcing (1,1,1) makes R=2, r=1 look like a sphere with a dent.
TORUS_VIEW = dict(elev=24, azim=-90)
TORUS_ASPECT = (1.0, 1.0 / 3.0, 1.0)
#: The box aspect above flattens the drawing inside a square axes box, which
#: leaves a lot of dead space; zoom scales the drawing up to fill it.
TORUS_ZOOM = 1.75


def _field_figure(problem: Problem, fields, titles, diverging: bool, path: str,
                  suptitle: str, value_label: str = "") -> None:
    """A field on the torus and in the (xi, eta) parameter rectangle.

    Two rows of the same quantity. The torus row is the literal picture the
    specification asks for; the parameter row is the same field drawn without
    occlusion, which is what makes fine structure legible. Both rows share one
    colour scale, and the scale is shared across every panel of the figure, so a
    panel that looks quiet really is quiet.
    """
    n = len(fields)
    fig = plt.figure(figsize=(2.9 * n, 5.2))
    X, Y, Z = torus_surface(problem)

    if diverging:
        cmap = style.diverging_cmap()
        vmax = max(float(np.abs(np.asarray(f, dtype=np.float64)).max()) for f in fields) or 1.0
        norm = style.diverging_norm(np.array([-vmax, vmax]))
    else:
        cmap = style.sequential_cmap()
        vmax = max(float(np.max(f)) for f in fields)
        vmin = min(float(np.min(f)) for f in fields)
        norm = plt.Normalize(vmin, vmax)

    for j, (f, title) in enumerate(zip(fields, titles)):
        _, _, Vg = lattice_grid(problem, np.asarray(f, dtype=np.float64))

        ax = fig.add_subplot(2, n, j + 1, projection="3d")
        ax.plot_surface(X, Y, Z, facecolors=cmap(norm(Vg)), rstride=1, cstride=1,
                        shade=False, antialiased=False, linewidth=0)
        ax.set_axis_off()
        ax.set_box_aspect(TORUS_ASPECT, zoom=TORUS_ZOOM)
        ax.view_init(**TORUS_VIEW)
        ax.set_title(title, fontsize=8.5, color=style.INK, pad=2)

        ax2 = fig.add_subplot(2, n, n + j + 1)
        im = ax2.imshow(Vg, origin="lower", cmap=cmap, norm=norm,
                        extent=[-np.pi, np.pi, -np.pi, np.pi], aspect="equal",
                        interpolation="nearest")
        ax2.set_xticks([-np.pi, 0, np.pi])
        ax2.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
        ax2.set_yticks([-np.pi, 0, np.pi])
        ax2.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
        ax2.set_xlabel(r"$\xi$")
        if j == 0:
            ax2.set_ylabel(r"$\eta$")
        ax2.grid(False)
        style.style_axes(ax2)
        if j == n - 1:
            cb = fig.colorbar(im, ax=fig.axes[n + j], fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=7, length=2)
            cb.outline.set_visible(False)
            if value_label:
                cb.set_label(value_label, fontsize=7.5)

    fig.suptitle(suptitle, fontsize=10, fontweight="semibold", color=style.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, path)


def plot_hard_errors(problem: Problem, selected: Dict[str, object], path: str,
                     n_show: int = 3) -> None:
    """The Jacobi-hard errors, next to kappa_epsilon on the same torus.

    The question the figure exists to help answer is spatial: do the error
    structures a point smoother cannot reduce sit on the low-kappa regions? The
    coefficient is drawn first so the comparison is visual and immediate, and the
    measured answer -- energy fraction inside the lowest-kappa quartile, for the
    hard set against the whole test set -- is written underneath. No causal
    reading is implied by the picture alone.
    """
    E_F = np.asarray(selected["E_F"])
    rho = np.asarray(selected["rho_jacobi"])
    n_show = min(n_show, E_F.shape[0])

    # The coefficient is drawn on its own scale (it is a magnitude, and its range
    # has nothing to do with the errors'), so it gets its own small figure rather
    # than sharing the diverging scale of the error fields.
    fields = [E_F[i] for i in range(n_show)]
    titles = [r"hard error #%d   $\rho_J = %.3f$" % (i + 1, rho[i])
              for i in range(n_show)]
    _field_figure(
        problem, fields, titles, diverging=True, path=path,
        suptitle="The %d test errors damped Jacobi reduces worst\n"
                 "area-weighted share of $e_F$ energy in the low-$\\kappa$ "
                 "quartile: %.0f%% for these, %.0f%% for the whole test set"
                 % (n_show,
                    100 * float(selected["error_energy_in_low_kappa_hard"]),
                    100 * float(selected["error_energy_in_low_kappa_all"])))


def plot_before_after(problem: Problem, selected: Dict[str, object], which: int,
                      path: str, deeponet_key: str = "deeponet_skip") -> None:
    """One hard error before and after each smoother, on a shared colour scale.

    The four panels share one diverging scale on purpose: separate scales would
    make the smallest residual look as loud as the original error, which is
    exactly the impression the figure must not give.
    """
    E_F = np.asarray(selected["E_F"])
    e = E_F[which]
    panels = [
        (e, r"original $e_F$   $\rho = 1$"),
        (np.asarray(selected["S_jacobi"])[which], r"after Jacobi   $\rho_J = %.3f$"
         % float(np.asarray(selected["rho_jacobi"])[which])),
        (np.asarray(selected["S_gs"])[which], r"after Gauss-Seidel   $\rho_{GS} = %.3f$"
         % float(np.asarray(selected["rho_gs"])[which])),
        (np.asarray(selected["S_" + deeponet_key])[which],
         r"after DeepONet   $\rho_\theta = %.3f$"
         % float(np.asarray(selected["rho_" + deeponet_key])[which])),
    ]

    _field_figure(
        problem, [p[0] for p in panels], [p[1] for p in panels],
        diverging=True, path=path,
        suptitle="Hard sample #%d before and after one smoothing step\n"
                 "shared colour scale, neutral grey at zero, so a quiet panel "
                 "really is quiet" % (which + 1))


# ---------------------------------------------------------------------------
# Supporting figures
# ---------------------------------------------------------------------------


def plot_training(histories: Dict[Tuple[float, str], List[Dict[str, object]]],
                  path: str, key: str = "val_loss") -> None:
    """The Stage-I validation loss against epoch, one curve per (epsilon, variant).

    The two encodings do different jobs here, so they use different channels.
    Epsilon is a *magnitude*, so it takes the single-hue sequential ramp, dark =
    small epsilon = high contrast. The architecture variant is an *identity*, so
    it takes the dash pattern: dotted for the plain MLP, solid for the one with
    the Jacobi skip. Using categorical hues for epsilon, or reusing a hue for the
    two variants, would both have been wrong for the same reason -- the channel
    would no longer mean one thing.
    """
    eps_values = sorted({e for e, _ in histories})
    ramp = style.sequential_cmap()
    fig, ax = plt.subplots(figsize=(7.4, 4.2))

    for (eps, arch), hist in sorted(histories.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        ep = [h["epoch"] for h in hist]
        y = [h[key] for h in hist]
        idx = eps_values.index(eps)
        frac = 0.15 + 0.75 * (idx / max(1, len(eps_values) - 1)) if len(eps_values) > 1 else 0.6
        c = ramp(frac)
        ls = style.VARIANT_DASH.get(arch, "-")
        ax.semilogy(ep, y, color=c, lw=1.9, ls=ls)
        style.label_line(ax, ep[-1], y[-1],
                         r"$\varepsilon=10^{%d}$ %s"
                         % (int(round(np.log10(eps))),
                            {"plain": "plain", "skip": "+skip",
                             "skip_fourier": "+skip fourier"}.get(arch, arch)),
                         c, fontsize=7)

    ax.set_xlabel("epoch")
    ax.set_ylabel(r"validation $\mathcal{L}_\mathrm{smooth}$")
    ax.set_title("Stage-I training: relative $A$-energy left after one smoothing step\n"
                 "dotted = plain MLP,  solid = with the learned Jacobi skip",
                 loc="left", fontsize=9.5)
    style.style_axes(ax)
    ax.margins(x=0.26)
    fig.tight_layout()
    _save(fig, path)


def plot_variant_bars(
    summary: Dict[str, Dict[str, Dict[str, float]]],
    methods: Sequence[str],
    section: str,
    metric: str,
    path: str,
    xlabel: str,
    title: str,
) -> None:
    """Mean reduction per method, one panel per epsilon, as horizontal bars.

    The form is length, not colour: every bar is the same single hue and identity
    comes from the y-axis label. That is what lets six methods sit on one figure
    without a sixth categorical slot -- the palette cap applies to charts where
    hue *is* the identity, and here it is not. Jacobi is marked by a reference
    line rather than by a colour, so a reader can see at a glance which variants
    at least match the classical baseline.
    """
    tags = sorted(summary.keys(), key=lambda t: -float(t))
    n = len(tags)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 0.42 * len(methods) + 2.3),
                             sharex=True)
    axes = np.atleast_1d(axes)
    cmap = style.sequential_cmap()
    y = np.arange(len(methods))[::-1]

    for j, tag in enumerate(tags):
        ax = axes[j]
        vals = [summary[tag].get(section, {}).get(m, {}).get(metric, np.nan)
                for m in methods]
        jac = summary[tag].get(section, {}).get("jacobi", {}).get(metric, np.nan)
        ax.barh(y, vals, height=0.62, color=cmap(0.62), zorder=3,
                edgecolor=style.SURFACE, linewidth=1.2)
        if np.isfinite(jac):
            ax.axvline(jac, color=style.INK, lw=1.4, ls="--", zorder=4)
            ax.annotate("Jacobi", xy=(jac, len(methods) - 0.55), xytext=(3, 0),
                        textcoords="offset points", fontsize=7,
                        color=style.INK_SECONDARY)
        for yi, v in zip(y, vals):
            if np.isfinite(v):
                ax.annotate("%.3f" % v, xy=(v, yi), xytext=(3, 0),
                            textcoords="offset points", va="center",
                            fontsize=7, color=style.INK_SECONDARY)
        ax.set_yticks(y)
        if j == 0:
            ax.set_yticklabels([style.METHOD_LABEL.get(m, m) for m in methods],
                               fontsize=7.5)
        else:
            ax.set_yticklabels([])
        ax.set_title(r"$\varepsilon = 10^{%d}$" % int(round(np.log10(float(tag)))),
                     fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylim(-0.7, len(methods) - 0.3)
        style.style_axes(ax)
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.22)

    fig.suptitle(title, fontsize=10, fontweight="semibold", color=style.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, path)


def plot_rho_distribution(rho_by_method: Dict[str, np.ndarray], path: str,
                          title: str) -> None:
    """Per-sample reduction distributions as a horizontal strip chart.

    A mean cannot distinguish "reduces everything a little" from "reduces most
    things a lot and amplifies a few", and that difference is the whole question
    for a smoother. Every sample is drawn, so the tail past 1 is visible.
    """
    names = [m for m in CURVE_METHODS if m in rho_by_method]
    fig, ax = plt.subplots(figsize=(7.0, 0.75 * len(names) + 1.9))
    rng = np.random.default_rng(0)

    for row, meth in enumerate(names):
        r = np.asarray(rho_by_method[meth], dtype=np.float64)
        c = style.METHOD_COLOR[meth]
        jitter = rng.uniform(-0.16, 0.16, size=r.size)
        ax.scatter(r, np.full(r.size, row) + jitter, s=7, color=c, alpha=0.45,
                   linewidths=0, zorder=3)
        ax.plot([np.mean(r)], [row], marker="|", ms=16, mew=2.2,
                color=style.INK, zorder=4)
        ax.annotate("mean %.3f" % float(np.mean(r)), xy=(np.mean(r), row + 0.34),
                    fontsize=7.5, color=style.INK_SECONDARY, ha="center")

    ax.axvline(1.0, color=style.INK_MUTED, lw=1.2, ls="--", zorder=1)
    ax.annotate("no reduction", xy=(1.0, len(names) - 0.45), xytext=(3, 0),
                textcoords="offset points", fontsize=7.5, color=style.INK_MUTED)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([style.METHOD_LABEL[m] for m in names])
    ax.set_xlabel(r"$\rho_F(e_F) = \|S(e_F)\|_A / \|e_F\|_A$  (one step)")
    ax.set_title(title, loc="left")
    ax.set_ylim(-0.7, len(names) - 0.2)
    style.style_axes(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, path)


def plot_fem_convergence(tables: Dict[str, List[Dict[str, float]]], path: str) -> None:
    """Manufactured-solution convergence, one panel per norm, one curve per epsilon.

    Plotted on log-log axes against the mesh size h ~ 1/sqrt(n_dof), with the
    reference slopes O(h) for the H1 seminorm and O(h^2) for L2 drawn alongside.
    A coefficient contrast of 1e4 that destroyed the convergence rate would show
    up here as a curve that bends away from its slope, and this is the check that
    has to pass before any smoother result is interpreted.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    norms = [("l2_error", r"$\|u-u_h\|_{L^2}$", 2.0),
             ("h1_error", r"$|u-u_h|_{H^1}$", 1.0),
             ("linf_error", r"$\|u-u_h\|_{L^\infty}$", 1.0)]

    for ax, (key, label, slope) in zip(axes, norms):
        for i, (tag, rows) in enumerate(sorted(tables.items())):
            c = style.CATEGORICAL[i % 4]
            h = np.array([1.0 / np.sqrt(r["dofs"]) for r in rows])
            y = np.array([r[key] for r in rows])
            ax.loglog(h, y, color=c, marker="o", markeredgecolor=style.SURFACE,
                      markeredgewidth=1.0, lw=1.6,
                      label=r"$\varepsilon=10^{%d}$" % int(round(np.log10(float(tag)))))
        h0 = np.array([1.0 / np.sqrt(16), 1.0 / np.sqrt(16384)])
        ax.loglog(h0, 3.0 * h0 ** slope, color=style.INK_MUTED, lw=1.0, ls="--")
        ax.annotate(r"$O(h^{%d})$" % int(slope), xy=(h0[-1], 3.0 * h0[-1] ** slope),
                    xytext=(4, 2), textcoords="offset points", fontsize=8,
                    color=style.INK_MUTED)
        ax.set_xlabel(r"mesh size $h \sim 1/\sqrt{n_\mathrm{dof}}$")
        ax.set_ylabel(label)
        ax.invert_xaxis()
        style.style_axes(ax)

    axes[0].legend(loc="upper left", fontsize=7.5)
    fig.suptitle("Manufactured-solution verification: $f_\\varepsilon$ recomputed "
                 "from $\\kappa_\\varepsilon$ at every $\\varepsilon$",
                 fontsize=10.5, fontweight="semibold", color=style.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, path)
