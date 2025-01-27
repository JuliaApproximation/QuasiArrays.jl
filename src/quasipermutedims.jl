# This file is based on a part of Julia. License is MIT: https://julialang.org/license

import Base: permutedims, permutedims!
export PermutedDimsQuasiArray

# Some day we will want storage-order-aware iteration, so put perm in the parameters
struct PermutedDimsQuasiArray{T,N,perm,iperm,AA<:AbstractQuasiArray} <: AbstractQuasiArray{T,N}
    parent::AA

    function PermutedDimsQuasiArray{T,N,perm,iperm,AA}(data::AA) where {T,N,perm,iperm,AA<:AbstractQuasiArray}
        (isa(perm, NTuple{N,Int}) && isa(iperm, NTuple{N,Int})) || error("perm and iperm must both be NTuple{$N,Int}")
        isperm(perm) || throw(ArgumentError(string(perm, " is not a valid permutation of dimensions 1:", N)))
        all(map(d->iperm[perm[d]]==d, 1:N)) || throw(ArgumentError(string(perm, " and ", iperm, " must be inverses")))
        new(data)
    end
end

"""
    PermutedDimsQuasiArray(A, perm) -> B

Given an AbstractQuasiArray `A`, create a view `B` such that the
dimensions appear to be permuted. Similar to `permutedims`, except
that no copying occurs (`B` shares storage with `A`).

See also: [`permutedims`](@ref).

# Examples
```jldoctest
julia> A = rand(3,5,4);

julia> B = PermutedDimsQuasiArray(A, (3,1,2));

julia> size(B)
(4, 3, 5)

julia> B[3,1,2] == A[1,2,3]
true
```
"""
function PermutedDimsQuasiArray(data::AbstractQuasiArray{T,N}, perm) where {T,N}
    length(perm) == N || throw(ArgumentError(string(perm, " is not a valid permutation of dimensions 1:", N)))
    iperm = invperm(perm)
    PermutedDimsQuasiArray{T,N,(perm...,),(iperm...,),typeof(data)}(data)
end

Base.parent(A::PermutedDimsQuasiArray) = A.parent
Base.size(A::PermutedDimsQuasiArray{T,N,perm}) where {T,N,perm} = genperm(size(parent(A)), perm)
Base.axes(A::PermutedDimsQuasiArray{T,N,perm}) where {T,N,perm} = genperm(axes(parent(A)), perm)

Base.similar(A::PermutedDimsQuasiArray, T::Type, dims::Base.Dims) = similar(parent(A), T, dims)


@inline function _getindex(::Type{IND}, A::PermutedDimsQuasiArray{T,N,perm,iperm}, I::IND) where {T,N,perm,iperm,IND}
    @boundscheck checkbounds(A, I...)
    @inbounds val = getindex(A.parent, genperm(I, iperm)...)
    val
end
@inline function _setindex!(::Type{IND}, A::PermutedDimsQuasiArray{T,N,perm,iperm}, val, I::IND) where {T,N,perm,iperm,IND}
    @boundscheck checkbounds(A, I...)
    @inbounds setindex!(A.parent, val, genperm(I, iperm)...)
    val
end

@inline genperm(I::NTuple{N,Any}, perm::Dims{N}) where {N} = ntuple(d -> I[perm[d]], Val(N))
@inline genperm(I, perm::AbstractVector{Int}) = genperm(I, (perm...,))

# function permutedims(A::AbstractQuasiArray, perm)
#     dest = similar(A, genperm(axes(A), perm))
#     permutedims!(dest, A, perm)
# end

permutedims(A::AbstractQuasiMatrix) = PermutedDimsQuasiArray(A, (2,1))
permutedims(v::AbstractQuasiVecOrMat{<:Real}) = v'
permutedims(v::AbstractQuasiVecOrMat{<:Number}) = transpose(v)
# permutedims(v::AbstractQuasiVector) = reshape(v, (1, length(v)))

# function Base.showarg(io::IO, A::PermutedDimsQuasiArray{T,N,perm}, toplevel) where {T,N,perm}
#     print(io, "PermutedDimsQuasiArray(")
#     Base.showarg(io, parent(A), false)
#     print(io, ", ", perm, ')')
#     toplevel && print(io, " with eltype ", eltype(A))
# end


## diagonal special case
permutedims(D::Diagonal{<:Any,<:SubArray{<:Any,1,<:AbstractQuasiArray}}) = D