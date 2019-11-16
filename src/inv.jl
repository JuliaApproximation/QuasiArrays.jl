
ApplyStyle(::typeof(\), ::Type{A}, ::Type{B}) where {A<:AbstractQuasiArray,B<:AbstractQuasiArray} = 
    quasildivapplystyle(MemoryLayout(A), MemoryLayout(B))
ApplyStyle(::typeof(\), ::Type{A}, ::Type{B}) where {A<:AbstractQuasiArray,B<:AbstractArray} = 
    quasildivapplystyle(MemoryLayout(A), MemoryLayout(B))
ApplyStyle(::typeof(\), ::Type{A}, ::Type{B}) where {A<:AbstractArray,B<:AbstractQuasiArray} = 
    quasildivapplystyle(MemoryLayout(A), MemoryLayout(B))

quasildivapplystyle(_, _) = LdivApplyStyle()

similar(L::Ldiv, ::Type{T}, axes::Tuple{<:AbstractQuasiVector,<:AbstractQuasiVector}) where T = similar(QuasiArray{T}, axes)
similar(L::Ldiv, ::Type{T}, axes::Tuple{<:AbstractQuasiVector,<:AbstractVector}) where T = similar(QuasiArray{T}, axes)
similar(L::Ldiv, ::Type{T}, axes::Tuple{<:AbstractVector,<:AbstractQuasiVector}) where T = similar(QuasiArray{T}, axes)
similar(L::Ldiv, ::Type{T}, axes::Tuple{<:AbstractQuasiVector}) where T = similar(QuasiArray{T}, axes)


function copyto!(dest::QuasiArray, L::Ldiv{QuasiArrayLayout,QuasiArrayLayout}) 
    copyto!(parent(dest), Ldiv(Array(L.A), Array(L.B)))
    dest
end
function copyto!(dest::QuasiArray, L::Ldiv{QuasiArrayLayout}) 
    copyto!(parent(dest), Ldiv(Array(L.A), L.B))
    dest
end
function copyto!(dest::QuasiArray, L::Ldiv{<:Any,QuasiArrayLayout}) 
    copyto!(parent(dest), Ldiv(L.A, Array(L.B)))
    dest
end
copyto!(dest::AbstractArray, L::Ldiv{QuasiArrayLayout,QuasiArrayLayout}) =
    copyto!(dest, Ldiv(Array(L.A), Array(L.B)))
copyto!(dest::AbstractArray, L::Ldiv{QuasiArrayLayout}) =
    copyto!(dest, Ldiv(Array(L.A), L.B))
copyto!(dest::AbstractArray, L::Ldiv{<:Any,QuasiArrayLayout}) =
    copyto!(dest, Ldiv(L.A, Array(L.B)))



for op in (:pinv, :inv)
    @eval ApplyStyle(::typeof($op), args::Type{<:AbstractQuasiArray}) = QuasiArrayApplyStyle()
end
## PInvQuasiMatrix


const PInvQuasiMatrix{T, ARGS} = ApplyQuasiMatrix{T,typeof(pinv),ARGS}
const InvQuasiMatrix{T, ARGS} = ApplyQuasiMatrix{T,typeof(inv),ARGS}

parent(A::PInvQuasiMatrix) = first(A.args)
parent(A::InvQuasiMatrix) = first(A.args)
axes(A::PInvQuasiMatrix) = axes(Applied(A))
size(A::PInvQuasiMatrix) = map(length, axes(A))
pinv(A::PInvQuasiMatrix) = first(A.args)

@propagate_inbounds getindex(A::PInvQuasiMatrix{T}, k::Int, j::Int) where T =
    (Applied(A)*[Zeros(j-1); one(T); Zeros(size(A,2) - j)])[k]

*(A::PInvQuasiMatrix, B::AbstractQuasiMatrix, C...) = apply(*,Applied(A), B, C...)
*(A::PInvQuasiMatrix, B::MulQuasiArray, C...) = apply(*,Applied(A), Applied(B), C...)

## QuasiArray special case
inv(A::QuasiMatrix) = QuasiArray(inv(A.parent), reverse(A.axes))