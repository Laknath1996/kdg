#!/bin/sh

# Gaussian parity
cd gxor
python run_exp_server.py -kdnversion 1 -c 2 -k 1e-5 -reps 45
cd ..

# Spiral
cd spiral
python run_exp_server.py -kdnversion 1 -c 3 -k 1e-5 -reps 45
cd ..

# Ellipse
cd ellipse
python run_exp_server.py -kdnversion 1 -c 3 -k 1e-5 -reps 45
cd ..

# Sinewave
cd sinewave
python run_exp_server.py -kdnversion 1 -c 5 -k 1e-6 -reps 45
cd ..

# Polynomial
cd polynomial
python run_exp_server.py -kdnversion 1 -c 3 -k 1e-5 -reps 45
cd ..