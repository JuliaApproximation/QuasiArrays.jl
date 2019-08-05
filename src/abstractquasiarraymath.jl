# This file is a part of Julia. License is MIT: https://julialang.org/license

 ## Basic functions ##

isreal(x::AbstractQuasiArray) = all(isreal,x)
iszero(x::AbstractQuasiArray) = all(iszero,x)
isreal(x::AbstractQuasiArray{<:Real}) = true
all(::typeof(isinteger), ::AbstractQuasiArray{<:Integer}) = true

## Constructors ##


dropdims(A; dims) = _dropdims(A, dims)
function _dropdims(A::AbstractQuasiArray, dims::Dims)
    for i in 1:length(dims)
        1 <= dims[i] <= ndims(A) || throw(ArgumentError("dropped dims must be in range 1:ndims(A)"))
        length(axes(A, dims[i])) == 1 || throw(ArgumentError("dropped dims must all be size 1"))
        for j = 1:i-1
            dims[j] == dims[i] && throw(ArgumentError("dropped dims must be unique"))
        end
    end
    d = ()
    for i = 1:ndims(A)
        if !in(i, dims)
            d = tuple(d..., axes(A, i))
        end
    end
    reshape(A, d::typeof(_sub(axes(A), dims)))
end
_dropdims(A::AbstractQuasiArray, dim::Integer) = _dropdims(A, (Int(dim),))

## Unary operators ##

conj(x::AbstractQuasiArray{<:Real}) = x
conj!(x::AbstractQuasiArray{<:Real}) = x

real(x::AbstractQuasiArray{<:Real}) = x
imag(x::AbstractQuasiArray{<:Real}) = zero(x)

+(x::AbstractQuasiArray{<:Number}) = x
*(x::AbstractQuasiArray{<:Number,2}) = x

# index A[:,:,...,i,:,:,...] where "i" is in dimension "d"


@inline selectdim(A::AbstractQuasiArray, d::Integer, i) = _selectdim(A, d, i, setindex(map(Slice, axes(A)), i, d))
@noinline function _selectdim(A, d, i, idxs)
    d >= 1 || throw(ArgumentError("dimension must be ≥ 1"))
    nd = ndims(A)
    d > nd && (i == 1 || throw(BoundsError(A, (ntuple(k->Colon(),d-1)..., i))))
    return view(A, idxs...)
end
