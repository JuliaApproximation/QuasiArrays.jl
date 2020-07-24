struct QuasiArray{T,N,AXES<:NTuple{N,AbstractVector}} <: AbstractQuasiArray{T,N}
    parent::Array{T,N}
    axes::AXES

    function QuasiArray{T,N,AXES}(par::AbstractArray{T,N}, axes::AXES) where {T,N,AXES<:NTuple{N,AbstractVector}} 
        size(par) == length.(axes) || throw(ArgumentError("Axes must be compatible with parent dimensions"))
        new{T,N,AXES}(convert(Array{T,N},par),axes)
    end
end

const QuasiMatrix{T,AXES<:NTuple{2,AbstractVector}} = QuasiArray{T,2,AXES}
const QuasiVector{T,AXES<:Tuple{AbstractVector}} = QuasiArray{T,1,AXES}


QuasiArray{T,N}(::UndefInitializer, axes::NTuple{N,AbstractVector}) where {T,N} =
    QuasiArray(Array{T}(undef, map(length,axes)), axes)
QuasiArray{T,N}(::UndefInitializer, axes::Vararg{AbstractVector,N}) where {T,N} =
    QuasiArray{T,N}(undef, axes)
QuasiVector(::UndefInitializer, axes::AbstractVector) where T =
    QuasiArray(Vector(undef,length(axes)), (axes,))
QuasiMatrix(::UndefInitializer, ax1::AbstractVector, ax2::AbstractVector) where T =
    QuasiArray(Matrix(undef,length(ax1),length(ax2)), (ax1,ax2))

QuasiArray(par::AbstractArray{T,N}, axes::NTuple{N,AbstractVector}) where {T,N} = 
    QuasiArray{T,N,typeof(axes)}(par, axes)
QuasiMatrix(par::AbstractMatrix{T}, axes::NTuple{2,AbstractVector}) where T = 
    QuasiArray{T,2,typeof(axes)}(par, axes)
QuasiVector(par::AbstractVector{T}, axes::Tuple{AbstractVector}) where T = 
    QuasiArray{T,1,typeof(axes)}(par, axes)

QuasiVector(par::AbstractVector{T}, axes::AbstractArray) where {T} = 
    QuasiVector(par, (axes,))
    

QuasiArray(a::AbstractQuasiArray) = QuasiArray(Array(a), map(domain,axes(a)))
QuasiArray{T}(a::AbstractQuasiArray) where T = QuasiArray(Array{T}(a), map(domain,axes(a)))
QuasiArray{T,N}(a::AbstractQuasiArray{<:Any,N}) where {T,N} = QuasiArray(Array{T}(a), map(domain,axes(a)))
QuasiArray{T,N,AXES}(a::AbstractQuasiArray{<:Any,N}) where {T,N,AXES} = QuasiArray{T,N,AXES}(Array{T}(a), map(domain,axes(a)))
QuasiMatrix(a::AbstractQuasiMatrix) = QuasiArray(a)
QuasiVector(a::AbstractQuasiVector) = QuasiArray(a)


_inclusion(d::Slice) = d
_inclusion(d::OneTo) = d
_inclusion(d::IdentityUnitRange{<:Integer}) = d
_inclusion(d::AbstractUnitRange{<:Integer}) = Slice(d)
_inclusion(d) = Inclusion(d)

axes(A::QuasiArray) = _inclusion.(A.axes)
parent(A::QuasiArray) = A.parent

@propagate_inbounds @inline function _getindex(::Type{IND}, A::QuasiArray, I::IND) where IND
    @boundscheck checkbounds(A, I...)
    A.parent[findfirst.(isequal.(I), A.axes)...]
end

@propagate_inbounds @inline function _setindex!(::Type{IND}, A::QuasiArray, v, I::IND) where IND
    @boundscheck checkbounds(A, I...)
    @inbounds A.parent[findfirst.(isequal.(I), A.axes)...] = v
    A
end

convert(::Type{T}, a::AbstractQuasiArray) where {T<:QuasiArray} = a isa T ? a : T(a)

function _quasimatrix_pow(A, p)
    axes(A,1) == axes(A,2) || throw(DimensionMismatch("axes must match"))
    QuasiArray(A.parent^p, A.axes)
end

^(A::QuasiMatrix, p::Number) = _quasimatrix_pow(A, p)
^(A::QuasiMatrix, p::Integer) = _quasimatrix_pow(A, p)