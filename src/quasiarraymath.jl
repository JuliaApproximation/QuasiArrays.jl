# This file is a part of Julia. License is MIT: https://julialang.org/license

## Unary operators ##

conj!(A::AbstractQuasiArray{<:Number}) = (@inbounds broadcast!(conj, A, A); A)

for f in (:-, :conj, :real, :imag)
    @eval ($f)(A::AbstractQuasiArray) = broadcast_preserving_zero_d($f, A)
end


## Binary arithmetic operators ##

for f in (:+, :-)
    @eval function ($f)(A::AbstractQuasiArray, B::AbstractQuasiArray)
        promote_shape(A, B) # check size compatibility
        broadcast_preserving_zero_d($f, A, B)
    end
end

function +(A::QuasiArray, Bs::QuasiArray...)
    for B in Bs
        promote_shape(A, B) # check size compatibility
    end
    broadcast_preserving_zero_d(+, A, Bs...)
end

for f in (:/, :\, :*)
    if f != :/
        @eval ($f)(A::Number, B::AbstractQuasiArray) = broadcast_preserving_zero_d($f, A, B)
    end
    if f != :\
        @eval ($f)(A::AbstractQuasiArray, B::Number) = broadcast_preserving_zero_d($f, A, B)
    end
end

