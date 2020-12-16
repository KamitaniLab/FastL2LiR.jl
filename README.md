# FastL2LiR.jl

Julia implementation of [fastl2lir](https://github.com/KamitaniLab/PyFastL2LiR).

Kei Majima developed the original algorithm.
Soma Nonaka implemented the initial Julia version.
Shuntaro C. Aoki modified and refactored the original implementation.

## Installation

``` julia
(@v1.5) pkg> add https://github.com/KamitaniLab/FastL2LiR.jl
```

## Usage

``` julia
using FastL2LiR

model = fit(X, Y, alpha)       # Without feature selection
model = fit(X, Y, alpha, 100)  # With feature selection
```
