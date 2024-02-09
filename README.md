### Introduction

This code implements a nudging algorithm to recover the parameters of a two-dimensional linear dynamical system using the approach of [Carlson, Hudson, Larios](https://doi.org/10.1137/19M1248583).

These are preliminary numerical experiments for a research project performed in 2021 under [Dr. Vincent R Martinez](http://math.hunter.cuny.edu/vmartine/) in the Department of Mathematics and Statistics at Hunter College, City University of New York

### Platform information

**Platform:** osx-64

**Python:** 3.8.8

**Numpy:** 1.20.1

**Scipy:** 1.6.2

**Matplotlib:** 3.3.4

### Installation

To create the appropriate virtual environment, run

```
conda env create -f env.yaml
```

To install the command line utilities, run the following in your activated environment:

```
pip install -e .
```

### Code repository
```
linear_nudging
├── bin
│   ├── nudge              # Command line tools
├── src                    
│   ├── figures.py         # Generates relevant figures
│   ├── simulation.py      # Runs parameter estimation algorithm
├── README.md
├── env.yaml
└── setup.py
```
### Running the scripts

The algorithm can be run from the command line using one of the following commands:

```
nudge WHICH --delta DELTA --alpha ALPHA --beta BETA --gamma GAMMA --mu MU --relax T_R

nudge WHICH --trace TR --det DET --mu MU --relax T_R
```

Here, `WHICH` specifies which parameters are to be estimated. Current options are `diagonal` (estimating `DELTA` and `GAMMA`) or `off-diagonal` (estimating `ALPHA` and `BETA`). The values of the other parameters are assumed to be known

The first option should be used if you want to specify a particular parameter matrix. Note the entries `DELTA`, `ALPHA`, `BETA`, and `GAMMA` correspond to `A(1,1)`, `A(1,2)`, `A(2,1)`, and `A(2,2)`, respectively.

The second option specifies a trace and determinant, and the simulator randomly generates an integer-valued parameter matrix with those attributes. In both cases the user needs to specify a nudging parameter `MU` (the algorithm uses a multiple of the identity matrix `MU*Identity(2)`), and relaxation time `T_R`.

Running the simulation will only generate a static figure with plots of the system solution and errors in parameters, position, velocity, but there is a function within figures.py that will generate an animation of the system solution which can be useful, also it's fun to watch the nudging happenining in real time :)
