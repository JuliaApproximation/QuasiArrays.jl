struct QuasiArray{T,N,AXES<:Tuple} <: AbstractQuasiArray{T,N}
    parent::Array{T,N}
    axes::AXES

    function QuasiArray{T,N,AXES}(par::AbstractArray{T,N}, axes::AXES) where {T,N,AXES<:Tuple} 
        size(par) == length.(axes) || throw(ArgumentError("Axes must be compatible with parent dimensions"))
        new{T,N,AXES}(convert(Array{T,N},par),axes)
    end
end

const QuasiMatrix{T,AXES<:Tuple} = QuasiArray{T,2,AXES}
const QuasiVector{T,AXES<:Tuple} = QuasiArray{T,1,AXES}


QuasiArray{T,N}(::UndefInitializer, axes::NTuple{N,AbstractQuasiOrVector{<:Real}}) where {T,N} =
    QuasiArray(Array{T}(undef, map(length,axes)), axes)
QuasiArray{T,N}(::UndefInitializer, axes::Vararg{AbstractQuasiOrVector{<:Real},N}) where {T,N} =
    QuasiArray{T,N}(undef, axes)
QuasiVector(::UndefInitializer, axes::AbstractQuasiOrVector{<:Real}) where T =
    QuasiArray(Vector(undef,length(axes)), (axes,))
QuasiMatrix(::UndefInitializer, ax1::AbstractQuasiOrVector{<:Real}, ax2::AbstractQuasiOrVector{<:Real}) where T =
    QuasiArray(Matrix(undef,length(ax1),length(ax2)), (ax1,ax2))

QuasiArray(par::AbstractArray{T,N}, axes::NTuple{N,AbstractQuasiOrVector{<:Real}}) where {T,N} = 
    QuasiArray{T,N,typeof(axes)}(par, axes)
QuasiMatrix(par::AbstractMatrix{T}, axes::NTuple{2,AbstractQuasiOrVector{<:Real}}) where T = 
    QuasiArray{T,2,typeof(axes)}(par, axes)
QuasiVector(par::AbstractVector{T}, axes::Tuple{AbstractQuasiOrVector{<:Real}}) where T = 
    QuasiArray{T,1,typeof(axes)}(par, axes)

QuasiVector(par::AbstractVector{T}, axes::AbstractArray) where {T} = 
    QuasiVector(par, (axes,))

QuasiArray(par::AbstractArray, axes::Tuple) = QuasiArray(convert(Array, par), axes)

_inclusion(d::Slice) = d
_inclusion(d::OneTo) = d
_inclusion(d) = Inclusion(d)


axes(A::QuasiArray) = _inclusion.(A.axes)
parent(A::QuasiArray) = A.parent

@propagate_inbounds @inline function getindex(A::QuasiArray{<:Any,N}, I::Vararg{Real,N}) where N
    @boundscheck checkbounds(A, I...)
    A.parent[findfirst.(isequal.(I), A.axes)...]
end

@propagate_inbounds @inline function setindex!(A::QuasiArray{<:Any,N}, v, I::Vararg{Real,N}) where N
    @boundscheck checkbounds(A, I...)
    @inbounds A.parent[findfirst.(isequal.(I), A.axes)...] = v
    v
end