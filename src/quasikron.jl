


"""
    QuasiKron(A, B, C...)

is a lazy representation of a Kronecker product of the quasi-arrays A, B, C.... Unlike 
standard kronecker product, we also kronecker the axes, using tuples.
"""

struct QuasiKron{T, N, Args<:Tuple} <: AbstractQuasiArray{T, N}
    args::Args
end

QuasiKron{T,N}(a...) where {T,N} = QuasiKron{T,N,typeof(a)}(a)
QuasiKron{T}(a::AbstractQuasiOrVector...) where T = QuasiKron{mapreduce(eltype,promote_type,a),1}(a...)
QuasiKron{T}(a::AbstractQuasiOrMatrix...) where T = QuasiKron{mapreduce(eltype,promote_type,a),2}(a...)
QuasiKron(a...) = QuasiKron{mapreduce(eltype,promote_type,a)}(a...)


quasikron(a::Union{Inclusion,OneTo}...) = Inclusion(ProductDomain(map(domain,a)...))
quasikron(a...) = QuasiKron(a...)

_kronaxes(::Tuple{}...) = ()
_kronaxes(a::Tuple...) = (quasikron(map(first,a)...), _kronaxes(map(tail,a)...)...)
axes(K::QuasiKron) = _kronaxes(map(axes,K.args)...)

function _getindex(::Type{IND}, K::QuasiKron, i::IND) where IND
    @boundscheck checkbounds(K, i...)
    prod(getindex.(K.args,i...))
end


MemoryLayout(::Type{<:QuasiKron{<:Any, <:Any, <:Args}}) where Args = kronlayout(tuple_type_memorylayouts(Args)...)
kronlayout(_...) = QuasiLazyLayout()
cardinality(K::QuasiKron) = prod(cardinality, K.args)

struct ArrayQuasiVector{T, Arr} <: AbstractQuasiVector{T}
    array::Arr
end

ArrayQuasiVector(A::AbstractArray{T}) where T = ArrayQuasiVector{T, typeof(A)}(A)

axes(a::ArrayQuasiVector) = (quasikron(axes(a.array)...),)

function _getindex(::Type{IND}, K::ArrayQuasiVector, i::IND) where IND
    @boundscheck checkbounds(K, i...)
    K.array[first(i)...]
end

function _getindex(::Type{IND}, f::ApplyQuasiVector{<:Any, typeof(*), <:Tuple{QuasiKron,ArrayQuasiVector}}, ij::IND) where IND
    K,c = f.args
    A,B = K.args
    i,j = ij[1]
    transpose(A[i,:]) * c.array * B[j,:]
end