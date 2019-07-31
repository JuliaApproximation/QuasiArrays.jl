# QuasiArrays.jl
A package for representing quasi-arrays

[![Build Status](https://travis-ci.org/JuliaApproximation/QuasiArrays.jl.svg?branch=master)](https://travis-ci.org/JuliaApproximation/QuasiArrays.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/e6ssqmpu01umofng?svg=true)](https://ci.appveyor.com/project/dlfivefifty/quasiarrays-jl)
[![codecov](https://codecov.io/gh/JuliaApproximation/QuasiArrays.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaApproximation/QuasiArrays.jl)


A _quasi-array_ is an array with non-classical indexing, including possibly
continuous indexing. This packages implements quasi-arrays. For example, we
can create a quasi-array where the first index is float valued:
```julia
using QuasiArrays
A = QuasiArray(rand(5,4,3), (range(0,1; length=5), Base.OneTo(4), [2,3,6]))
A[0.25,2,6] # equivalent to parent(A)[2,2,3]
```
Analogues of many the base types are supported. For example, we can create a quasi-diagonal matrix
```julia
v = QuasiVector(rand(5), 0:0.5:2) # diagonal
D = QuasiDiagonal(v)
D[0.5,0.5] # equivalent to parent(D)[0.5] == parent(parent(D))[2]
```
We can take views of quasi-arrays:
```julia
view(A, 0:0.25:0.5, 2:3, [2,6])[2,1,2] # equivalent to A[0.25,2,6]
```
And we can also broadcast, which preserves axes:
```julia
exp.(v)[0.5] # equivalent to exp(v[0.5])
```

Finally, by combining with IntervalSets.jl we support continuous indexing:
```julia
using IntervalSets
x = Inclusion(0.0..1.0) # Inclusion is identity, e.g. x[0.2] == 0.2
D = QuasiDiagonal(x)
D[0.1,0.2] # 0.0
D[0.1,0.1] # 0.1
```
Full functionality for continuous quasi-arrays is in [ContinuumArrays.jl](https://github.com/JuliaApproximation/ContinuumArrays.jl).


## Relation to other Julia packages

There are other packages that allow non-standard indexing, such as
[NamedArrays](https://github.com/davidavdav/NamedArrays.jl) and [AxisArrays](https://github.com/JuliaArrays/AxisArrays.jl).
QuasiArrays.jl focusses on linear algebra aspects, that is, the axes of a quasi-array
encode the inner product. This forms the basis of [ContinuumArrays.jl](https://github.com/JuliaApproximation/ContinuumArrays.jl)
which is a fresh approach to finite element methods and spectral methods, where bases
are represented as quasi-matrices and discretizations arise from linear algebra
operations on quasi-matrices.
