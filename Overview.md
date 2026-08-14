# RCS Optimization via Coding Metasurface GA

A genetic algorithm that optimizes the per-element phase states of an
`N x M` coding metasurface to reduce monostatic radar cross-section (RCS),
evaluated with an array-factor model over a grid of theta/phi observation
angles.

## How it works

1. **Steering matrix** (`steering.py`) — builds the complex array-factor
   steering matrix for the `N x M` grid over the theta/phi angle sweep.
2. **Population** (`population.py`) — initializes the GA population with
   structured seeds (all-zero, checkerboard, row-alternating,
   column-alternating) plus random individuals.
3. **Fitness** (`fitness.py`) — decodes each individual's phase states,
   computes its array factor, and scores it as the RCS reduction in dB
   relative to a fully-reflecting PEC reference:
   `fitness = 10 * log10(PEC_peak / candidate_peak)`.
   Higher fitness = smaller scattered-power peak = better reduction.
4. **Genetic operators** (`genetic_operators.py`) — tournament selection,
   uniform crossover, adaptive mutation rate, and an elite-protected
   generational replacement scheme.
5. **Optimizer** (`optimizer.py`) — `optimize_rcs(...)` ties the above
   together into the full generational loop and returns the best pattern
   found.

## Install

```bash
pip install -r requirements.txt
```

## Run the example

```bash
python -m rcs_optimization.examples.run_optimization
```

This runs a 30x30, 1-bit (2-state) array over 200 generations and prints
the optimized coding matrix, RCS reduction in dB, and a comparison against
a plain checkerboard pattern.

## Use in your own code

```python
from rcs_optimization import optimize_rcs

result = optimize_rcs(N=30, M=30, B=1, pop_size=200, max_generations=200)

print(result["best_fitness_dB"])
print(result["best_individual"].reshape(result["N"], result["M"]))
```

## Repo layout

```
rcs_optimization/
├── __init__.py
├── steering.py
├── population.py
├── fitness.py
├── genetic_operators.py
├── optimizer.py
└── examples/
    ├── __init__.py
    └── run_optimization.py
requirements.txt
README.md
```
