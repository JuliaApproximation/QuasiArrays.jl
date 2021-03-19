struct QuasiArray{T,N,AXES<:NTuple{N,AbstractVector}} <: AbstractQuasiArray{T,N}
    parent::Array{T,N}
    axes::AXES

    function QuasiArray{T,N,AXES}(par::AbstractArray{T,N}, axes::AXES) where {T,N,AXES<:NTuple{N,AbstractVector}} 
        size(par) == length.(axes) || throw(ArgumentError("Axes must be compatible with parent dimensions"))
        new{T,N,AXES}(convert(Array{T,N},par),axes)
    end
end

QuasiArray{T,N,AXES}(par::AbstractArray{<:Any,N}, axes::AXES) where {T,N,AXES<:NTuple{N,AbstractVector}} =
    QuasiArray{T,N,AXES}(convert(AbstractArray{T,N}, par)::AbstractArray{T,N}, axes)

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

QuasiArray{T}(par::AbstractArray{<:Any,N}, axes::NTuple{N,AbstractVector}) where {T,N} = 
    QuasiArray{T,N,typeof(axes)}(par, axes)
QuasiMatrix{T}(par::AbstractMatrix, axes::NTuple{2,AbstractVector}) where T = 
    QuasiMatrix{T,typeof(axes)}(par, axes)
QuasiVector{T}(par::AbstractVector, axes::Tuple{AbstractVector}) where T = 
    QuasiVector{T,typeof(axes)}(par, axes)

QuasiArray(par::AbstractArray{T,N}, axes::Vararg{AbstractVector,N}) where {T,N} = 
    QuasiArray(par, axes)
QuasiMatrix(par::AbstractMatrix{T}, axes::Vararg{AbstractVector,2}) where T = 
    QuasiMatrix(par, axes)
QuasiVector(par::AbstractVector{T}, axes::AbstractVector) where T = 
    QuasiVector(par, (axes,))

QuasiArray{T}(par::AbstractArray{<:Any,N}, axes::Vararg{AbstractVector,N}) where {T,N} = 
    QuasiArray{T}(par, axes)
QuasiMatrix{T}(par::AbstractMatrix, axes::Vararg{AbstractVector,2}) where T = 
    QuasiMatrix{T}(par, axes)
QuasiVector{T}(par::AbstractVector, axes::AbstractVector) where T = 
    QuasiVector{T}(par, (axes,))



QuasiMatrix(λ::UniformScaling, ax::NTuple{2,AbstractVector}) = QuasiMatrix(Matrix(λ,map(length,ax)...), ax)
QuasiMatrix{T}(λ::UniformScaling, ax::NTuple{2,AbstractVector}) where T = QuasiMatrix{T}(Matrix(λ,map(length,ax)...), ax)
QuasiMatrix(λ::UniformScaling, ax1::AbstractVector, ax2::AbstractVector) = QuasiMatrix(λ, (ax1, ax2))
QuasiMatrix{T}(λ::UniformScaling, ax1::AbstractVector, ax2::AbstractVector) where T = QuasiMatrix{T}(λ, (ax1, ax2))


QuasiArray(par::AbstractArray) = QuasiArray(par, axes(par))
QuasiMatrix(par::AbstractArray) = QuasiMatrix(par, axes(par))
QuasiVector(par::AbstractArray) = QuasiVector(par, axes(par))

QuasiArray{T,N,AXES}(par::AbstractArray{T,N}, axes::NTuple{N,AbstractQuasiOrVector}) where {T,N,AXES} =
    QuasiArray{T,N,AXES}(par, map(domain,axes))
QuasiArray{T,N}(par::AbstractArray{T,N}, axes::NTuple{N,AbstractQuasiOrVector}) where {T,N} =
    QuasiArray{T,N}(par, map(domain,axes))
QuasiArray{T}(par::AbstractArray{T,N}, axes::NTuple{N,AbstractQuasiOrVector}) where {T,N} =
    QuasiArray{T}(par, map(domain,axes))
QuasiArray(par::AbstractArray{T,N}, axes::NTuple{N,AbstractQuasiOrVector}) where {T,N} = 
    QuasiArray(par, map(domain,axes))
QuasiMatrix(par::AbstractMatrix{T}, axes::NTuple{2,AbstractQuasiOrVector}) where T = 
    QuasiMatrix(par, map(domain,axes))
QuasiVector(par::AbstractVector{T}, axes::Tuple{AbstractQuasiOrVector}) where T = 
    QuasiVector(par, map(domain,axes))

QuasiVector(par::AbstractVector{T}, axes::AbstractArray) where {T} = 
    QuasiVector(par, (axes,))
    

QuasiArray(a::AbstractQuasiArray) = QuasiArray(a[map(collect,axes(a))...], axes(a))
QuasiArray{T}(a::AbstractQuasiArray) where T = QuasiArray(Array{T}(a), axes(a))
QuasiArray{T,N}(a::AbstractQuasiArray{<:Any,N}) where {T,N} = QuasiArray(Array{T}(a), axes(a))
QuasiArray{T,N,AXES}(a::AbstractQuasiArray{<:Any,N}) where {T,N,AXES} = QuasiArray{T,N,AXES}(Array{T}(a), axes(a))
QuasiMatrix(a::AbstractQuasiMatrix) = QuasiArray(a)
QuasiVector(a::AbstractQuasiVector) = QuasiArray(a)


convert(::Type{T}, a::AbstractQuasiArray) where {T<:QuasiArray} = a isa T ? a : T(a)
convert(::Type{QuasiArray{T,N}}, a::AbstractQuasiArray) where {T,N} = a isa QuasiArray{T,N} ? a : QuasiArray{T,N}(a)
convert(::Type{QuasiArray{T}}, a::AbstractQuasiArray) where T = a isa QuasiArray{T} ? a : QuasiArray{T}(a)
convert(::Type{AbstractQuasiArray{T,N}}, a::QuasiArray) where {T,N} = convert(QuasiArray{T,N}, a)
convert(::Type{AbstractQuasiArray{T}}, a::QuasiArray) where T = convert(QuasiArray{T}, a)
convert(::Type{AbstractQuasiArray{T,N}}, a::QuasiArray{T,N}) where {T,N} = a
convert(::Type{AbstractQuasiArray{T}}, a::QuasiArray{T}) where T = a


_inclusion(d::Slice) = d
_inclusion(d::OneTo) = d
_inclusion(d::IdentityUnitRange{<:Integer}) = d
_inclusion(d::AbstractUnitRange{<:Integer}) = IdentityUnitRange(d)
_inclusion(d) = Inclusion(d)

axes(A::QuasiArray) = map(_inclusion,A.axes)
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

function _quasimatrix_pow(A, p)
    axes(A,1) == axes(A,2) || throw(DimensionMismatch("axes must match"))
    QuasiArray(A.parent^p, A.axes)
end

^(A::QuasiMatrix, p::Number) = _quasimatrix_pow(A, p)
^(A::QuasiMatrix, p::Integer) = _quasimatrix_pow(A, p)


==(A::QuasiArray{T,N,NTuple{N,OneTo{Int}}}, B::AbstractArray{V,N}) where {T,V,N} =
    axes(A) == axes(B) && A.parent == B
==(B::AbstractArray{V,N}, A::QuasiArray{T,N,NTuple{N,OneTo{Int}}}) where {T,V,N} =
    A == B
