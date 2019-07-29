# QuasiArrays.jl
A package for representing quasi-arrays


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