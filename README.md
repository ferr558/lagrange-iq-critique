# Underdetermination of Finite Sequences and Construct Validity in Intelligence Testing

This repository contains a mathematical and psychometric paper on a simple but
important claim:

> A finite numerical sequence does not, by itself, determine a unique next term
> unless additional assumptions are imposed on the class of admissible rules or
> on the criterion used to select among them.

The project combines a formal argument based on Lagrange interpolation with a
computational visualization and a broader discussion of what sequence-completion
items may actually measure in intelligence testing.

## Repository Contents

- `paper.tex`: Italian LaTeX source of the paper
- `paper.pdf`: compiled Italian PDF
- `paper_eng.tex`: English LaTeX source
- `paper_eng.pdf`: compiled English PDF
- `lagrange.cu`: CUDA implementation used to generate families of interpolating polynomials
- `plot.py`: Python script that visualizes the solution space
- `output.csv`: generated envelope of sampled interpolants
- `points.csv`: sampled input points used by the plotting pipeline
- `lagrange_plot.png`: final figure included in the paper

## What the Paper Argues

The paper develops three connected points:

1. A finite sequence is mathematically underdetermined unless extra constraints
   are added.
2. The commonly expected answer in sequence items is usually the rule intended
   by the test designer, not the only logically possible continuation.
3. This matters for construct validity: an item may still be psychometrically
   useful, correlate with other tests, and contribute to the estimation of the
   `g` factor, while remaining conceptually narrower than the label
   "general intelligence" may suggest.

The paper does **not** argue that intelligence tests are worthless. Its claim is
more precise: the interpretation of sequence-completion items requires more
theoretical care than is often made explicit.

## The `g` Factor

The paper includes an extended, non-technical explanation of:

- what the `g` factor is
- why it was introduced in psychometrics
- how it is estimated from correlations among multiple cognitive tasks
- why `g` is a latent statistical construct rather than a directly observed quantity

Both the Italian and English versions are written to remain understandable to
readers without prior training in psychometrics or higher mathematics.

## Build the Papers

If you have a LaTeX environment with `latexmk` installed:

```bash
latexmk -pdf paper.tex
latexmk -pdf paper_eng.tex
```

## Reproduce the Figure

The visualization pipeline requires CUDA and a Python environment with
`numpy`, `pandas`, `matplotlib`, and `sympy`.

Compile and run the CUDA program:

```bash
nvcc -O2 -o lagrange lagrange.cu
./lagrange
```

Then generate the figure:

```bash
python3 plot.py
```

This produces:

- `output.csv`
- `points.csv`
- `lagrange_plot.png`

## Why Two Papers?

- `paper.tex` / `paper.pdf`: full Italian version
- `paper_eng.tex` / `paper_eng.pdf`: full English version

The English file is not a short abstract or summary. It is a complete translation
of the full paper.

## Suggested Citation

If you want to reference this repository, cite the paper title and link to the
repository. If you plan to submit the article formally, replace the placeholder
author metadata in the LaTeX sources first.

## Status

This repository is currently structured as a self-contained research note with:

- a full paper in two languages
- a computational demonstration
- a reproducible figure-generation pipeline

It can be used as:

- a standalone argument
- a draft for formal submission
- a public research repository accompanying the paper
