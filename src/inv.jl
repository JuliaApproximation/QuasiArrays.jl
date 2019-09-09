ApplyStyle(::typeof(\), ::Type{<:AbstractQuasiArray}, ::Type{<:AbstractQuasiArray}) = QuasiArrayApplyStyle()
ApplyStyle(::typeof(\), ::Type{<:AbstractQuasiArray}, ::Type{<:AbstractArray}) = QuasiArrayApplyStyle()
ApplyStyle(::typeof(\), ::Type{<:AbstractArray}, ::Type{<:AbstractQuasiArray}) = QuasiArrayApplyStyle()    

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
