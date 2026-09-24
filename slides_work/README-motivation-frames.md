# Polished motivation frames (chapter 1)

Four frame fragments for `Masterthesis_Fan Yang_14th`. Each file is a
`\frametitle`-only fragment, matching the convention of `rublkm`'s `\load`
(`\load{dir}` = `\begin{frame}[t]\input{dir/slide.tex}\end{frame}`), so they
drop into `_Directory/` unchanged. None of them define macros.

## What each frame does

| Folder | Frame title | Role |
| --- | --- | --- |
| `Ch1_ResearchMotivation` | Research Motivation | **New.** The three-field Venn (Engineering / Numerics / CS meeting at PDE on curved surfaces) beside the four-question loop. |
| `Ch1_Motivation` | Motivation: PDEs on Curved Surfaces | **Rewritten** (was 4 long bullets + block). Now two columns: the problem and its two competing demands, then the classical answer; the "where it becomes difficult" block spans the foot. |
| `Ch1_TwoPipelines` | Classical Solvers vs.\ Neural Operators | **New.** The two pipelines side by side, headers centred over each column, and the block that binds them to the same loop. |
| `Ch1_PipelineComparison` | The Same Four Questions, Two Pipelines | **New, optional.** Table answering all four questions for both pipelines. Drop it if the talk is tight — it overlaps `Ch1_OperatorLearning`. |

## The narrative spine

The four questions are the same ones asked throughout, and they are the reason
the two pipelines are comparable at all:

1. Can I solve it?
2. Can I measure the error? (and so say how far the result can be believed)
3. Can I speed it up?
4. Have I used every effort available for a practical problem?

Question 4 is where the learned smoother enters — which is what
`Ch1_Smoother` and `Ch1_OperatorLearning` then develop. The loop is stated
informally here and made precise as the six research questions in
`Ch1_Questions`; the frames say so explicitly so the two numbering schemes do
not read as a contradiction.

## Wiring in `main.tex`

Replace the chapter-1 load block with:

```latex
\load{_Directory/MiniOutline}
\load{_Directory/Ch1_ResearchMotivation}   % NEW
\load{_Directory/Ch1_Motivation}           % rewritten
\load{_Directory/Ch1_Smoother}
\load{_Directory/Ch1_TwoPipelines}         % NEW
\load{_Directory/Ch1_OperatorLearning}
\load{_Directory/Ch1_PipelineComparison}   % NEW (optional)
\load{_Directory/Ch1_Questions}
\load{_Directory/Ch1_Structure}
```

## Build status

Verified with the project's own preamble and `rublkm` style, pdflatex
(TeX Live 2024): **exit 0, no LaTeX errors, no `Overfull \vbox` on any of the
four frames.** Two things worth knowing if you edit them:

- Use `\begin{columns}[T]`, not `[t]`. Lowercase `[t]` aligns *first
  baselines*; because a `tikzpicture` sits on its own bottom edge, that drops
  the second column to the foot of the first and overflows the slide by ~5cm.
- `text width` in a TikZ node must be generous at this class size (20pt).
  `\small` here is roughly 17pt, so a 42mm box wraps `Discretisation` onto its
  own line and boxes at fixed `at (x,y)` coordinates then overlap. The
  fragments use 70mm boxes with `below=8mm of` chaining so spacing follows
  natural box height.
