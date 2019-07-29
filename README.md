# QuasiArrays.jl
A package for representing quasi-arrays

[![Build Status](https://travis-ci.org/JuliaApproximation/QuasiArrays.jl.svg?branch=master)](https://travis-ci.org/JuliaApproximation/QuasiArrays.jl)
[![Coverage Status](https://coveralls.io/repos/github/JuliaApproximation/QuasiArrays.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaApproximation/QuasiArrays.jl?branch=master)


A _quasi-array_ is an array with non-classical indexing, including possibly 
continuous indexing. This packages implements quasi-arrays. For example, we 
can create a quasi-array where the first index is float valued:
```julia
A = QuasiArray(rand(5,4,3), (range(0,1; length=5), Base.OneTo(4), [2,3,6]))
A[0.25,2,6] # equivalent to parent(A)[2,2,3]
```
Many of the base types are supported. For example, we can create a quasi-diagonal matrix
```julia
v = QuasiArray(rand(5), (range(0,1; length=5),)) # diagonal
D = QuasiDiagonal(v)
D[0.25,0.25] # equivalent to parent(D)[0.25] == parent(parent(D))[2]
```

## Relation to other Julia packages

There are other packages that allow non-standard indexing, such as
[NamedArrays](https://github.com/davidavdav/NamedArrays.jl) and [AxisArrays](https://github.com/JuliaArrays/AxisArrays.jl).
QuasiArrays.jl focusses on linear algebra aspects, that is, the axes of a quasi-array
encode the inner product. This forms the basis of [ContinuumArrays.jl](https://github.com/JuliaApproximation/ContinuumArrays.jl)
which is a fresh approach to finite element methods and spectral methods, where bases
are represented as quasi-matrices and discretizations arise from linear algebra
operations on quasi-matrices. 