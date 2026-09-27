# The learned smoother — slides of the 15th-meeting status report

A beamer deck built from the article project **`MS_Fanyang_Report`**
(`6ab97e78f1eb520f9e954b58`, *"The learned smoother for geometric multigrid on
surfaces — Status report, 15th meeting"*, 16 pp).

It uses the **same RUB/LKM template and the same one-folder-per-frame pattern as
`MS_Fanyang_15th`**: `_stylefiles/rublkm.sty` + `defdb1.sty`, each frame a
`_Directory/<Frame>/slide.tex` pulled in by `\load{...}`, logos in `_figures/`.

## Compile

From this directory:

```bash
pdflatex main.tex
pdflatex main.tex      # second pass for the page numbers / TOC
```

No bibtex: the reference list is a literal `thebibliography` in
`_Directory/Reference/slide.tex`, so `\cite` resolves directly.

## Page map — 17 pages, mirroring the report's six sections

| # | Frame | Report section |
|---|-------|----------------|
| 1 | `preface` (title) | — |
| 2 | `Outline` | — |
| 3 | `A1_Question` | §1 Introduction — the working solver, the question, attribution, the three stages |
| 4 | `B1_Model` | §2 The model problem — torus, PDE, manufactured solution, discretisation |
| 5 | `C1_TwoLevel` | §3.1–3.2 hierarchy, transfers, Galerkin, the two-level step, `C_h` |
| 6 | `C2_Projection` | §3.2 Prop. — `C_h = Π_F` exactly, and `E_TG = S_post Π_F S_pre` |
| 7 | `C3_Split` | §3.3–3.5 `e = e_C + e_F`, the three identities, the direct sum |
| 8 | `C4_LearnedStep` | §3.6–3.7 where the network enters; one-sided vs symmetric; the leak λ |
| 9 | `D1_Objectives` | §4.1 the three objectives |
| 10 | `D2_Loss` | §4.2–4.4 the objective used, the constraints, what is left out |
| 11 | `E1_Classical` | §5.1 the classical reference (and the constraint it imposes) |
| 12 | `E2_Setting` | §5.2 the Stage I setting, and the asserted identities |
| 13 | `E3_Diagnostics` | §5.3–5.4 trunk floor and rank-*k* floor |
| 14 | `E4_Measurement` | §5.5 the table and the answer |
| 15 | `E5_Defect` | §5.6–5.7 per mechanism, and the defect |
| 16 | `F1_Conclusion` | §6 established / next steps / protocol |
| 17 | `Reference` | bibliography |

Two report sections were deliberately folded rather than given their own frame to
keep the deck near the requested length: §1.2–1.3 (the stage table and "what makes
the question well posed") now sits at the foot of page 3, and §2.1–2.4 (the
discretisation table) is a block on page 4.

## Verified, not transcribed

Every number on every slide is taken from a table already in the report — the
classical reference and convergence orders, the two floor tables, the headline
Stage I table, the per-mechanism table and the defect table. **No number was
invented or recomputed for the deck.** The μ_C = 1.0400 / 1.0928 pair that appears
without its own table on page 10 is the width-256/768 row of the defect table on
page 15.

## Build checks

* `pdflatex` ×2 → **17 pages, exit 0, no errors**
* **0 `Overfull \vbox`**
* no `Overfull \hbox` other than the **25.61 pt per frame that is this template's
  baseline** (the `\hspace*{0.9cm}` in the frametitle) — compare per frame, not the total
* **no ink below the green footer rule on any page** (`_src/footer_check.py` in the
  parent tree; it compares each page against the fixed footer, which is identical on
  every page except the title slide, which has none). This is the check that matters:
  beamer logs no `\vbox` warning for a frame whose text merely runs under the rule.

## Template notes that look like bugs but are not

* `_stylefiles/rublkm.sty` does `\addtolength{\parskip}{3mm}` — this is what makes
  dense frames overflow, and beamer stays silent about it.
* `\alert` is RUB green **and one size larger**
  (`\setbeamerfont{alerted text}{size=\larger[1]}`), so it costs vertical space.
* The page geometry is `279.4mm × 215.9mm` (old slide format), so `\textwidth` is
  **250 mm** — tables sized for an A4 text width come out cramped in the top-left.
* `\frametitle` adds `\hspace*{0.9cm}`, hence the 25.61 pt hbox on *every* content
  frame.
