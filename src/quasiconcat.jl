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