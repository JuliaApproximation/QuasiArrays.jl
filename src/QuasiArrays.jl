module QuasiArrays
using Base, LinearAlgebra, LazyArrays
import Base: getindex, size, axes, length, ==, isequal, iterate, CartesianIndices, LinearIndices,
                Indices, IndexStyle, getindex, setindex!, parent, vec, convert, similar, copy, copyto!, zero,
                map, eachindex, eltype, first, last, firstindex, lastindex, in, reshape, all,
                isreal, iszero, isempty, empty, isapprox, fill!, getproperty
import Base: @_inline_meta, DimOrInd, OneTo, @_propagate_inbounds_meta, @_noinline_meta,
                DimsInteger, error_if_canonical_getindex, @propagate_inbounds, _return_type,
                _maybetail, tail, _getindex, _maybe_reshape, index_ndims, _unsafe_getindex,
                index_shape, to_shape, unsafe_length, @nloops, @ncall, unalias,
                to_index, to_indices, _to_subscript_indices, _splatmap, dataids
import Base: ViewIndex, Slice, IdentityUnitRange, ScalarIndex, RangeIndex, view, viewindexing, ensure_indexable, index_dimsum,
                check_parent_index_match, reindex, _isdisjoint, unsafe_indices, _unsafe_ind2sub,
                _ind2sub, _sub2ind, _ind2sub_recurse, _lookup,
                parentindices, reverse, ndims, checkbounds,
                promote_shape, maybeview, checkindex, checkbounds_indices,
                throw_boundserror, rdims, replace_in_print_matrix
import Base: *, /, \, +, -, inv
import Base: exp, log, sqrt,
          cos, sin, tan, csc, sec, cot,
          cosh, sinh, tanh, csch, sech, coth,
          acos, asin, atan, acsc, asec, acot,
          acosh, asinh, atanh, acsch, asech, acoth
import Base: Array, Matrix, Vector

import Base.Broadcast: materialize, materialize!, BroadcastStyle, AbstractArrayStyle, Style, broadcasted, Broadcasted, Unknown,
                        newindex, broadcastable, preprocess, _eachindex, _broadcast_getindex,
                        DefaultArrayStyle, axistype, throwdm, instantiate, combine_eltypes, eltypes                   

import LinearAlgebra: transpose, adjoint, checkeltype_adjoint, checkeltype_transpose, Diagonal,
                        AbstractTriangular, pinv, inv, promote_leaf_eltypes

import LazyArrays: MemoryLayout, UnknownLayout, Mul, ApplyLayout, ⋆,
                    lmaterialize, _lmaterialize, InvOrPInv, ApplyStyle, LazyLayout, FlattenMulStyle,
                    Applied, flatten, _flatten,
                    rowsupport, colsupport, tuple_type_memorylayouts,
                    LdivApplyStyle, most,
                    _mul, rowsupport, DiagonalLayout, adjointlayout, transposelayout, conjlayout

import Base.IteratorsMD

export AbstractQuasiArray, AbstractQuasiMatrix, AbstractQuasiVector, materialize,
       QuasiArray, QuasiMatrix, QuasiVector, QuasiDiagonal, Inclusion,
       QuasiAdjoint, QuasiTranspose, ApplyQuasiArray, ApplyQuasiMatrix, ApplyQuasiVector,
       BroadcastQuasiArray, BroadcastQuasiMatrix, BroadcastQuasiVector

if VERSION < v"1.3-"
    """
    broadcast_preserving_zero_d(f, As...)

    Like [`broadcast`](@ref), except in the case of a 0-dimensional result where it returns a 0-dimensional container

    Broadcast automatically unwraps zero-dimensional results to be just the element itself,
    but in some cases it is necessary to always return a container — even in the 0-dimensional case.
    """
    function broadcast_preserving_zero_d(f, As...)
        bc = broadcasted(f, As...)
        r = materialize(bc)
        return length(axes(bc)) == 0 ? fill!(similar(bc, typeof(r)), r) : r
    end
    broadcast_preserving_zero_d(f) = fill(f())
    broadcast_preserving_zero_d(f, as::Number...) = fill(f(as...))
else
    import Base.Broadcast: broadcast_preserving_zero_d
end

abstract type AbstractQuasiArray{T,N} end
AbstractQuasiVector{T} = AbstractQuasiArray{T,1}
AbstractQuasiMatrix{T} = AbstractQuasiArray{T,2}
AbstractQuasiVecOrMat{T} = Union{AbstractQuasiVector{T}, AbstractQuasiMatrix{T}}
const AbstractQuasiOrVector{T} = Union{AbstractVector{T},AbstractQuasiVector{T}}
const AbstractQuasiOrMatrix{T} = Union{AbstractMatrix{T},AbstractQuasiMatrix{T}}
const AbstractQuasiOrArray{T} = Union{AbstractArray{T},AbstractQuasiArray{T}}


cardinality(d) = length(d)

size(A::AbstractQuasiArray) = map(cardinality, axes(A))
axes(A::AbstractQuasiArray) = error("Override axes for $(typeof(A))")

include("indices.jl")
include("abstractquasiarray.jl")
include("multidimensional.jl")
include("subquasiarray.jl")
include("quasireshapedarray.jl")
include("quasibroadcast.jl")
include("abstractquasiarraymath.jl")

include("quasiarray.jl")
include("quasiarraymath.jl")

include("lazyquasiarrays.jl")

include("matmul.jl")
include("inv.jl")
include("quasiadjtrans.jl")
include("quasidiagonal.jl")

promote_leaf_eltypes(x::AbstractQuasiArray{T}) where {T<:Number} = T
promote_leaf_eltypes(x::AbstractQuasiArray) = mapreduce(promote_leaf_eltypes, promote_type, x; init=Bool)


function isapprox(x::AbstractQuasiArray, y::AbstractQuasiArray;
    atol::Real=0,
    rtol::Real=Base.rtoldefault(promote_leaf_eltypes(x),promote_leaf_eltypes(y),atol),
    nans::Bool=false, norm::Function=norm)
    d = norm(x - y)
    if isfinite(d)
        return d <= max(atol, rtol*max(norm(x), norm(y)))
    else
        # Fall back to a component-wise approximate comparison
        return all(ab -> isapprox(ab[1], ab[2]; rtol=rtol, atol=atol, nans=nans), zip(x, y))
    end
end

end
