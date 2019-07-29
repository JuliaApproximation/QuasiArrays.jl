struct QuasiArray{T,N,AXES<:Tuple} <: AbstractQuasiArray{T,N}
    parent::Array{T,N}
    axes::AXES

    function QuasiArray{T,N,AXES}(par::Array{T,N}, axes::AXES) where {T,N,AXES<:Tuple} 
        size(par) == length.(axes) || throw(ArgumentError("Axes must be compatible with parent dimensions"))
        new{T,N,AXES}(par,axes)
    end
end

const QuasiMatrix{T,AXES<:Tuple} = QuasiArray{T,2,AXES}
const QuasiVector{T,AXES<:Tuple} = QuasiArray{T,1,AXES}

QuasiArray(par::Array{T,N}, axes::AXES) where {T,N,AXES<:Tuple} = 
    QuasiArray{T,N,AXES}(par, axes)
QuasiMatrix(par::Matrix{T}, axes::AXES) where {T,AXES<:Tuple} = 
    QuasiArray{T,2,AXES}(par, axes)
QuasiVector(par::Vector{T}, axes::AXES) where {T,AXES<:Tuple} = 
    QuasiArray{T,1,AXES}(par, axes)

QuasiVector(par::Vector{T}, axes::AbstractArray) where {T} = 
    QuasiVector(par, (axes,))

QuasiArray(par::AbstractArray, axes::Tuple) = QuasiArray(convert(Array, par), axes)

axes(A::QuasiArray) = Inclusion.(A.axes)
parent(A::QuasiArray) = A.parent

@propagate_inbounds @inline function getindex(A::QuasiArray{<:Any,N}, I::Vararg{Real,N}) where N
    @boundscheck checkbounds(A, I...)
    A.parent[findfirst.(isequal.(I), A.axes)...]
end