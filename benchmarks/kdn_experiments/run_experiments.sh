#!/bin/sh

## Gaussian Parity
python gaussian_parity_exp.py -c 1 -k 1e-5 -reps 100

# ## Spiral
python spiral_exp.py -c 3 -k 1e-5 -reps 100

# ## Ellipse
python ellipse_exp.py -c 3 -k 1e-5 -reps 100

# ## Gaussian Sparse Parity
python gaussian_sparse_parity_exp.py -c 2 -k 1e-5 -reps 100

# ## Step
python step_exp.py -c 3 -k 1e-5 -reps 100

# ## Polynomial
python polynomial_exp.py -c 2 -k 1e-5 -reps 100

# ## sinewave
python sinewave_exp.py -c 2 -k 1e-5 -reps 100
