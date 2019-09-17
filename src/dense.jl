# This file is based on a part of Julia. License is MIT: https://julialang.org/license


# function tr(A::AbstractQuasiMatrix{T}) where T
#     n = checksquare(A)
#     t = zero(T)
#     for i=1:n
#         t += A[i,i]
#     end
#     t
# end


# function kron(a::AbstractMatrix{T}, b::AbstractMatrix{S}) where {T,S}
#     require_one_based_indexing(a, b)
#     R = Matrix{promote_op(*,T,S)}(undef, size(a,1)*size(b,1), size(a,2)*size(b,2))
#     m = 0
#     @inbounds for j = 1:size(a,2), l = 1:size(b,2), i = 1:size(a,1)
#         aij = a[i,j]
#         for k = 1:size(b,1)
#             R[m += 1] = aij*b[k,l]
#         end
#     end
#     R
# end

# kron(a::Number, b::Union{Number, AbstractVecOrMat}) = a * b
# kron(a::AbstractVecOrMat, b::Number) = a * b
# kron(a::AbstractVector, b::AbstractVector) = vec(kron(reshape(a ,length(a), 1), reshape(b, length(b), 1)))
# kron(a::AbstractMatrix, b::AbstractVector) = kron(a, reshape(b, length(b), 1))
# kron(a::AbstractVector, b::AbstractMatrix) = kron(reshape(a, length(a), 1), b)

# Matrix power
(^)(A::AbstractQuasiMatrix, p::Integer) = 
    if p == 1 
        copy(A) 
    elseif p < 0 
        inv(A)^(-p) 
    else
        A*A^(p-1)
    end
function (^)(A::AbstractQuasiMatrix{T}, p::Integer) where T<:Integer
    # make sure that e.g. [1 1;1 0]^big(3)
    # gets promotes in a similar way as 2^big(3)
    TT = promote_op(^, T, typeof(p))
    return convert(AbstractQuasiMatrix{TT}, A)^p
end
# function integerpow(A::AbstractQuasiMatrix{T}, p) where T
#     TT = promote_op(^, T, typeof(p))
#     return (TT == T ? A : copyto!(similar(A, TT), A))^Integer(p)
# end

# function (^)(A::AbstractQuasiMatrix{T}, p::Real) where T
#     n = checksquare(A)

#     # Quicker return if A is diagonal
#     if isdiag(A)
#         TT = promote_op(^, T, typeof(p))
#         retmat = copy_oftype(A, TT)
#         for i in 1:n
#             retmat[i, i] = retmat[i, i] ^ p
#         end
#         return retmat
#     end

#     # For integer powers, use power_by_squaring
#     isinteger(p) && return integerpow(A, p)

#     # If possible, use diagonalization
#     if issymmetric(A)
#         return (Symmetric(A)^p)
#     end
#     if ishermitian(A)
#         return (Hermitian(A)^p)
#     end

#     # Otherwise, use Schur decomposition
#     return schurpow(A, p)
# end

# (^)(A::AbstractQuasiMatrix, p::Number) = exp(p*log(A))
