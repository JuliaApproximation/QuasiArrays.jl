###
# Hcat
####
hcat(A::AbstractQuasiArray...) = apply(hcat, A...)

ApplyStyle(::typeof(hcat), ::Type{<:AbstractQuasiArray}...) = LazyQuasiArrayApplyStyle()

axes(f::ApplyQuasiMatrix{<:Any,typeof(hcat)}) = (axes(f.args[1],1), Base.OneTo(sum(size.(f.args,2))))

function getindex(f::ApplyQuasiMatrix{T,typeof(hcat)}, k::Number, j::Number) where T
    ξ = j
    for A in f.args
        n = size(A,2)
        ξ ≤ n && return T(A[k,ξ])::T
        ξ -= n
    end
    throw(BoundsError(f, (k,j)))
end

"""
    InclusionUnion(A, B, C...)

is used to represent the union of (A...)
"""

struct InclusionUnion{T, Args<:Tuple} <: AbstractInclusion{T}
    args::Args
end

InclusionUnion{T}(a...) where T = InclusionUnion{T,typeof(a)}(a)
InclusionUnion(a...) = InclusionUnion{mapreduce(eltype,promote_type,a)}(a...)

copy(d::InclusionUnion) = d

==(A::InclusionUnion, B::InclusionUnion) = A.args == B.args
axes(S::InclusionUnion) = (S,)
unsafe_indices(S::InclusionUnion) = (S,)
axes1(S::InclusionUnion) = S

first(S::InclusionUnion) = first(first(S.args))
last(S::InclusionUnion) = last(last(S.args))
size(S::InclusionUnion) = (sum(size.(S.args,1)),)
length(S::InclusionUnion) = sum(map(length,S.args))
cardinality(S::InclusionUnion) = sum(map(cardinality,S.args))
show(io::IO, r::InclusionUnion) = print(io, "InclusionUnion(", r.args, ")")
# iterate(S::InclusionUnion, s...) = iterate(S.domain, s...)

in(x, S::InclusionUnion) = any(in.(Ref(x), S.args))

union(x::AbstractInclusion...) = InclusionUnion(x...)

"""
    UnionVcat

is an analogue of `Vcat` that takes the union of the axes.
If there is overlap, it uses the first in order.
"""
struct UnionVcat{T,N,Args<:Tuple} <: AbstractQuasiArray{T,N}
    args::Args
end

function UnionVcat{T,2}(A...) where T
    ncols = axes(A[1], 2)
    for j = 2:length(A)
        if axes(A[j], 2) != ncols
            throw(ArgumentError("axes of columns of each array must match (got $(map(x->axes(x,2), A)))"))
        end
    end
    UnionVcat{T,2,typeof(A)}(A)
end

UnionVcat{T,1}(A...) where T = UnionVcat{T,1,typeof(A)}(A)

UnionVcat{T}(a::AbstractQuasiOrVector...) where T = UnionVcat{T,1}(a...)
UnionVcat{T}(a::AbstractQuasiOrMatrix...) where T = UnionVcat{T,2}(a...)
UnionVcat(a...) = UnionVcat{mapreduce(eltype,promote_type,a)}(a...)

axes(A::UnionVcat{<:Any,1}) = (union(axes.(A.args,1)...),)
axes(A::UnionVcat{<:Any,2}) = (union(axes.(A.args,1)...),axes(A.args[1],2))

function _getindex(::Type{IND}, A::UnionVcat{T,1}, (x,)::IND) where {IND,T}
    for a in A.args
        x in axes(a,1) && return convert(T,a[x])::T
    end
    throw(BoundsError(A, I))
end

function _getindex(::Type{IND}, A::UnionVcat{T,2}, (x,j)::IND) where {IND,T}
    for a in A.args
        x in axes(a,1) && return convert(T,a[x,j])::T
    end
    throw(BoundsError(A, I))
end