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
QuasiMatrix(par::Array{T,N}, axes::AXES) where {T,N,AXES<:Tuple} = 
    QuasiMatrix{T,N,AXES}(par, axes)
QuasiVector(par::Array{T,N}, axes::AXES) where {T,N,AXES<:Tuple} = 
    QuasiVector{T,N,AXES}(par, axes)

axes(A::QuasiArray) = Inclusion.(A.axes)
parent(A::QuasiArray) = A.parent

@propagate_inbounds @inline function getindex(A::QuasiArray, I::Real...) 
    @boundscheck checkbounds(A, I...)
    A.parent[findfirst.(isequal.(I), A.axes)...]
end