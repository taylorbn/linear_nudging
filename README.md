### Introduction

This code implements a nudging algorithm to recover the parameters of a two-dimensional linear dynamical system using the approach of [Carlson, Hudson, Larios](https://doi.org/10.1137/19M1248583).

This is the beginning stages of a research project performed under [Dr. Vincent R Martinez](http://math.hunter.cuny.edu/vmartine/) in the Department of Mathematics and Statistics at Hunter College, City University of New York.

### Platform information

**Platform:** osx-64

**Python:** 3.8.8

**Numpy:** 1.20.1

**Scipy:** 1.6.2

**Matplotlib:** 3.3.4

### Package installation

To install the required packages, type in terminal:

```
pip install --user PACKAGE_NAME
```

To ensure reproducibility of results, use the same package versions as above.

```
pip install --user numpy==1.20.1
pip install --user scipy==1.6.2
pip install --user matplotlib==3.3.4
```

Alternatively using Anaconda, you can use the specification file provided to create an identical conda environment to run the scripts by typing in terminal:

```
conda create --name myenv --file spec-file.txt
```

This spec was created on the osx-64 platorm, and may not work correctly on others. See [documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for more details.

### Code repository

This folder contains two python scripts for running the simulation and generating the figures:
* **linear_nudging.py:** Main script to run the algorithm
* **figures.py:** Generates figures

### Running the scripts

To run the provided scripts, navigate to the relevant folder by typing in terminal:

```
cd FOLDER_NAME
```

The algorithm can be run from the command line using one of the following commands:

```
python linear_nudging.py DELTA ALPHA BETA GAMMA MU T_R
python linear_nudging.py TR DET MU T_R
```

The first option should be used if you want to specify a particular parameter matrix. Note the entries DELTA, ALPHA, BETA, GAMMA correspond to A(1,1), A(1,2), A(2,1), A(2,2). The second option specifies a trace and determinant, and the program randomly generates an integer-valued parameter matrix with those attributes. In both cases the user needs to specify a nudging parameter MU (the algorithm uses a multiple of the identity matrix MU*Identity(2)), and relaxation time T_R.

The code is set up to only update parameters with initial guesses different from the true value. At the moment this is adjusted within the script.

Running the simulation will only generate a static figure with plots of the system solution and errors in parameters, position, velocity, but there is a function within figures.py that will generate an animation of the system solution which can be useful, also it's fun to watch the nudging happenining in real time :)
