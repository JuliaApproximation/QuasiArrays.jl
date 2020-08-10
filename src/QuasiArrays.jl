module QuasiArrays
using Base, LinearAlgebra, LazyArrays, ArrayLayouts
import Base: getindex, size, axes, axes1, length, ==, isequal, iterate, CartesianIndices, LinearIndices,
                Indices, IndexStyle, getindex, setindex!, parent, vec, convert, similar, copy, copyto!, zero,
                map, eachindex, eltype, first, last, firstindex, lastindex, in, reshape, all,
                isreal, iszero, isempty, empty, isapprox, fill!, getproperty
import Base: @_inline_meta, DimOrInd, OneTo, @_propagate_inbounds_meta, @_noinline_meta,
                DimsInteger, error_if_canonical_getindex, @propagate_inbounds, _return_type,
                _maybetail, tail, _getindex, _maybe_reshape, index_ndims, _unsafe_getindex,
                index_shape, to_shape, unsafe_length, @nloops, @ncall, unalias, _unaliascopy,
                to_index, to_indices, _to_subscript_indices, _splatmap, dataids, 
                compute_stride1, compute_offset1, fill_to_length
import Base: ViewIndex, Slice, IdentityUnitRange, ScalarIndex, RangeIndex, view, viewindexing, ensure_indexable, index_dimsum,
                check_parent_index_match, reindex, _isdisjoint, unsafe_indices, _unsafe_ind2sub,
                _ind2sub, _sub2ind, _ind2sub_recurse, _lookup, SubArray,
                parentindices, reverse, ndims, checkbounds, uncolon,
                promote_shape, maybeview, checkindex, checkbounds_indices,
                throw_boundserror, rdims, replace_in_print_matrix, show,
                hcat, vcat, hvcat
import Base: *, /, \, +, -, ^, inv
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
                        AbstractTriangular, pinv, inv, promote_leaf_eltypes, power_by_squaring,
                        integerpow, schurpow, tr, factorize

import ArrayLayouts: indextype
import LazyArrays: MemoryLayout, UnknownLayout, Mul, ApplyLayout, BroadcastLayout, ⋆,
                    InvOrPInv, ApplyStyle, AbstractLazyLayout, LazyLayout, 
                    MulStyle, MulAddStyle, LazyArrayApplyStyle, combine_mul_styles, DefaultArrayApplyStyle,
                    Applied, flatten, _flatten, arguments, _mat_mul_arguments, _vec_mul_arguments,
                    rowsupport, colsupport, tuple_type_memorylayouts, applylayout, broadcastlayout,
                    LdivStyle, most, InvLayout, PInvLayout, sub_materialize, lazymaterialize,
                    _mul, rowsupport, DiagonalLayout, adjointlayout, transposelayout, conjlayout,
                    sublayout, call, LazyArrayStyle, layout_getindex

import Base.IteratorsMD

export AbstractQuasiArray, AbstractQuasiMatrix, AbstractQuasiVector, materialize,
       QuasiArray, QuasiMatrix, QuasiVector, QuasiDiagonal, Inclusion,
       QuasiAdjoint, QuasiTranspose, ApplyQuasiArray, ApplyQuasiMatrix, ApplyQuasiVector,
       BroadcastQuasiArray, BroadcastQuasiMatrix, BroadcastQuasiVector, indextype

import Base.Broadcast: broadcast_preserving_zero_d

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
include("quasiconcat.jl")

include("matmul.jl")
include("inv.jl")
include("quasiadjtrans.jl")
include("quasidiagonal.jl")
include("dense.jl")

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
