const QuasiArrayMulArray{p, q, T, V} =
    Applied{<:Any, typeof(*), <:Tuple{<:AbstractQuasiArray{T,p}, <:AbstractArray{V,q}}}

const ArrayMulQuasiArray{p, q, T, V} =
    Applied{<:Any, typeof(*), <:Tuple{<:AbstractArray{T,p}, <:AbstractQuasiArray{V,q}}}

const QuasiArrayMulQuasiArray{p, q, T, V} =
    Applied{<:Any, typeof(*), <:Tuple{<:AbstractQuasiArray{T,p}, <:AbstractQuasiArray{V,q}}}
####
# Matrix * Vector
####
const QuasiMatMulVec{T, V} = QuasiArrayMulArray{2, 1, T, V}
const QuasiMatMulMat{T, V} = QuasiArrayMulArray{2, 2, T, V}
const QuasiMatMulQuasiMat{T, V} = QuasiArrayMulQuasiArray{2, 2, T, V}


function getindex(M::Mul{<:AbstractQuasiArrayApplyStyle}, k::Number)
    A,Bs = first(M.args), tail(M.args)
    B = _mul(Bs...)
    ret = zero(eltype(M))
    for j = rowsupport(A, k) ∩ colsupport(B,1)
        ret += A[k,j] * B[j]
    end
    ret
end

function _mul_quasi_getindex(M::Mul, k::Number, j::Number)
    A,Bs = first(M.args), tail(M.args)
    B = _mul(Bs...)
    ret = zero(eltype(M))
    @inbounds for ℓ in (rowsupport(A,k) ∩ colsupport(B,j))
        ret += A[k,ℓ] * B[ℓ,j]
    end
    ret
end

getindex(M::Mul{<:AbstractQuasiArrayApplyStyle}, k::Number, j::Number) =
    _mul_quasi_getindex(M, k, j)

getindex(M::Mul{<:AbstractQuasiArrayApplyStyle}, k::Integer, j::Integer) =
    _mul_quasi_getindex(M, k, j)


function getindex(M::QuasiMatMulVec, k::AbstractArray)
    A,B = M.args
    ret = zeros(eltype(M),length(k))
    @inbounds for j in axes(A,2)
        ret .+= view(A,k,j) .* B[j]
    end
    ret
end


*(A::AbstractQuasiArray, B...) = fullmaterialize(apply(*,A,B...))
*(A::AbstractQuasiArray, B::AbstractQuasiArray, C...) = fullmaterialize(apply(*,A,B,C...))
*(A::AbstractArray, B::AbstractQuasiArray, C...) = fullmaterialize(apply(*,A,B,C...))

for op in (:pinv, :inv)
    @eval $op(A::AbstractQuasiArray) = fullmaterialize(apply($op,A))
end

axes(L::Ldiv{<:Any,<:Any,<:AbstractQuasiMatrix}) =
    (axes(L.A, 2),axes(L.B,2))
axes(L::Ldiv{<:Any,<:Any,<:AbstractQuasiVector}) =
    (axes(L.A, 2),)    

 \(A::AbstractQuasiArray, B::AbstractQuasiArray) = apply(\,A,B)


*(A::AbstractQuasiArray, B::Mul, C...) = apply(*,A, B.args..., C...)
*(A::Mul, B::AbstractQuasiArray, C...) = apply(*,A.args..., B, C...)


####
# MulQuasiArray
#####

const MulQuasiArray{T, N, Args<:Tuple} = ApplyQuasiArray{T, N, typeof(*), Args}

const MulQuasiVector{T, Args<:Tuple} = MulQuasiArray{T, 1, Args}
const MulQuasiMatrix{T, Args<:Tuple} = MulQuasiArray{T, 2, Args}

const Vec = MulQuasiVector


_ApplyArray(F, factors...) = ApplyQuasiArray(F, factors...)
_ApplyArray(F, factors::AbstractArray...) = ApplyArray(F, factors...)

MulQuasiOrArray = Union{MulArray,MulQuasiArray}

_factors(M::MulQuasiOrArray) = M.args
_factors(M) = (M,)

_flatten(A::MulQuasiArray, B...) = _flatten(Applied(A), B...)
flatten(A::MulQuasiArray) = ApplyQuasiArray(flatten(Applied(A)))


function fullmaterialize(M::Applied{<:Any,typeof(*)})
    M_mat = materialize(flatten(M))
    typeof(M_mat) <: MulQuasiOrArray || return M_mat
    typeof(Applied(M_mat)) == typeof(M) || return(fullmaterialize(M_mat))

    ABC = M_mat.args
    length(ABC) ≤ 2 && return flatten(M_mat)

    AB = most(ABC)
    Mhead = fullmaterialize(Mul(AB...))

    typeof(_factors(Mhead)) == typeof(AB) ||
        return fullmaterialize(Mul(_factors(Mhead)..., last(ABC)))

    BC = tail(ABC)
    Mtail =  fullmaterialize(Mul(BC...))
    typeof(_factors(Mtail)) == typeof(BC) ||
        return fullmaterialize(Mul(first(ABC), _factors(Mtail)...))

    apply(*,first(ABC), Mtail.args...)
end

fullmaterialize(M::ApplyQuasiArray) = flatten(fullmaterialize(Applied(M)))
fullmaterialize(M) = flatten(M)


adjoint(A::MulQuasiArray) = ApplyQuasiArray(*, reverse(adjoint.(A.args))...)
transpose(A::MulQuasiArray) = ApplyQuasiArray(*, reverse(transpose.(A.args))...)

function similar(A::MulQuasiArray)
    B,a = A.args
    B*similar(a)
end

function similar(A::QuasiArrayMulArray)
    B,a = A.args
    applied(*, B, similar(a))
end

function copy(a::MulQuasiArray)
    @_propagate_inbounds_meta
    copymutable(a)
end

function copyto!(dest::MulQuasiArray, src::MulQuasiArray)
    d = last(dest.args)
    s = last(src.args)
    copyto!(IndexStyle(d), d, IndexStyle(s), s)
    dest
end




_mul_tail_support(j, Z) = maximum(last.(colsupport.(Ref(Z),j)))
_mul_tail_support(j, Z, Y, X...) = _mul_tail_support(OneTo(_mul_tail_support(j,Z)), Y, X...)

function _mul_getindex(k, j, A, B) 
    M = min(_mul_tail_support(j,B), maximum(last.(rowsupport.(Ref(A),k))))
    A[k,1:M]*B[1:M,j]
end

function _mul_getindex(k, j, A, B, C, D...) 
    N = _mul_tail_support(j, reverse(D)..., C, B)
    M = min(maximum(last.(rowsupport.(Ref(A),k))), N)
    _mul_getindex(OneTo(N), j, A[k,OneTo(M)]*B[OneTo(M),OneTo(N)], C, D...)
end

getindex(A::MulQuasiMatrix, k::AbstractVector{<:Number}, j::AbstractVector{<:Number}) = 
    _mul_getindex(k, j, A.args...)
 


quasimulapplystyle(_...) = QuasiArrayApplyStyle()

ApplyStyle(::typeof(*), ::Type{A}, ::Type{B}, C::Type...) where {A<:AbstractQuasiArray,B<:Union{AbstractArray,AbstractQuasiArray}} = 
    quasimulapplystyle(MemoryLayout(A), MemoryLayout(B), MemoryLayout.(C)...)
ApplyStyle(::typeof(*), ::Type{A}, ::Type{B}, C::Type...) where {A<:AbstractArray,B<:AbstractQuasiArray} = 
    quasimulapplystyle(MemoryLayout(A), MemoryLayout(B), MemoryLayout.(C)...)
ApplyStyle(::typeof(*), ::Type{A}, ::Type{B}, ::Type{C}, D::Type...) where {A<:AbstractQuasiArray,B<:Union{AbstractArray,AbstractQuasiArray},C<:Union{AbstractArray,AbstractQuasiArray}} = 
    quasimulapplystyle(MemoryLayout(A), MemoryLayout(B), MemoryLayout(C), MemoryLayout.(D)...)
ApplyStyle(::typeof(*), ::Type{A}, ::Type{B}, ::Type{C}, D::Type...) where {A<:AbstractArray,B<:AbstractQuasiArray,C<:Union{AbstractArray,AbstractQuasiArray}} = 
    quasimulapplystyle(MemoryLayout(A), MemoryLayout(B), MemoryLayout(C), MemoryLayout.(D)...)
ApplyStyle(::typeof(*), ::Type{A}, ::Type{B}, ::Type{C}, D::Type...) where {A<:AbstractArray,B<:AbstractArray,C<:AbstractQuasiArray} = 
    quasimulapplystyle(MemoryLayout(A), MemoryLayout(B), MemoryLayout(C), MemoryLayout.(D)...)
ApplyStyle(::typeof(*), ::Type{A}, ::Type{B}, ::Type{C}, ::Type{D}, E::Type...) where {A<:AbstractQuasiArray,B<:Union{AbstractArray,AbstractQuasiArray},C<:Union{AbstractArray,AbstractQuasiArray},D<:Union{AbstractArray,AbstractQuasiArray}} = 
    quasimulapplystyle(MemoryLayout(A), MemoryLayout(B), MemoryLayout(C), MemoryLayout(D), MemoryLayout.(E)...)
ApplyStyle(::typeof(*), ::Type{A}, ::Type{B}, ::Type{C}, ::Type{D}, E::Type...) where {A<:AbstractArray,B<:AbstractQuasiArray,C<:Union{AbstractArray,AbstractQuasiArray},D<:Union{AbstractArray,AbstractQuasiArray}} = 
    quasimulapplystyle(MemoryLayout(A), MemoryLayout(B), MemoryLayout(C), MemoryLayout(D), MemoryLayout.(E)...)
ApplyStyle(::typeof(*), ::Type{A}, ::Type{B}, ::Type{C}, ::Type{D}, E::Type...) where {A<:AbstractArray,B<:AbstractArray,C<:AbstractQuasiArray,D<:Union{AbstractArray,AbstractQuasiArray}} = 
    quasimulapplystyle(MemoryLayout(A), MemoryLayout(B), MemoryLayout(C), MemoryLayout(D), MemoryLayout.(E)...)
ApplyStyle(::typeof(*), ::Type{A}, ::Type{B}, ::Type{C}, ::Type{D}, E::Type...) where {A<:AbstractArray,B<:AbstractArray,C<:AbstractArray,D<:AbstractQuasiArray} = 
    quasimulapplystyle(MemoryLayout(A), MemoryLayout(B), MemoryLayout(C), MemoryLayout(D), MemoryLayout.(E)...)    


####
# Matrix * Array
####



function _lmaterialize(A::MulQuasiArray, B, C...)
    As = A.args
    flatten(_ApplyArray(*, reverse(tail(reverse(As)))..., _lmaterialize(last(As), B, C...)))
end



function _rmaterialize(Z::MulQuasiArray, Y, W...)
    Zs = Z.args
    flatten(_ApplyArray(*, _rmaterialize(first(Zs), Y, W...), tail(Zs)...))
end


####
# Lazy \ ApplyArray. This applies to first arg.
#####

struct LmaterializeApplyStyle <: ApplyStyle end

quasimulapplystyle(::InvLayout, _) = LdivApplyStyle()
quasimulapplystyle(::PInvLayout, _) = LdivApplyStyle()
quasimulapplystyle(::InvLayout, ::LazyLayout) = LdivApplyStyle()
quasimulapplystyle(::PInvLayout, ::LazyLayout) = LdivApplyStyle()
quasimulapplystyle(::InvLayout, ::LazyLayout, _...) = LmaterializeApplyStyle()
quasimulapplystyle(::PInvLayout, ::LazyLayout, _...) = LmaterializeApplyStyle()
quasimulapplystyle(::ApplyLayout{typeof(*)}, ::ApplyLayout{typeof(*)}, _...) = FlattenMulStyle()
quasimulapplystyle(::ApplyLayout{typeof(*)}, _...) = FlattenMulStyle()
quasimulapplystyle(::ApplyLayout{typeof(*)}, ::LazyLayout, _...) = FlattenMulStyle()
quasimulapplystyle(::ApplyLayout{typeof(*)}, _, ::LazyLayout, _...) = FlattenMulStyle()
quasimulapplystyle(::ApplyLayout{typeof(*)}, ::LazyLayout, ::LazyLayout, _...) = FlattenMulStyle()


ApplyStyle(::typeof(\), ::Type{<:AbstractQuasiArray}, ::Type{<:Applied}) = LdivApplyStyle()
\(A::AbstractQuasiArray, B::Applied) = apply(\, A, B)
copy(L::Ldiv{LazyLayout,ApplyLayout{typeof(*)}}) = *(L.A \  first(L.B.args),  tail(L.B.args)...)
copy(L::Ldiv{LazyLayout,ApplyLayout{typeof(*)},<:Any,<:AbstractQuasiArray}) = L.A \  Applied(L.B)


copy(A::Applied{LmaterializeApplyStyle,typeof(*)}) = lmaterialize(A)