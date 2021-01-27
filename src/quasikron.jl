


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


quasikron(a::Inclusion...) = Inclusion(ProductDomain(map(domain,a)...))
quasikron(a...) = QuasiKron(a...)

_kronaxes(::Tuple{}...) = ()
_kronaxes(a::Tuple...) = (quasikron(map(first,a)...), _kronaxes(map(tail,a)...)...)
axes(K::QuasiKron) = _kronaxes(map(axes,K.args)...)

function getindex(K::QuasiKron{<:Any,1}, i::Tuple)
    @boundscheck checkbounds(K, i)
    prod(getindex.(K.args,i))
end