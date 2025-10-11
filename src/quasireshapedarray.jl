vec_layout(lay, _) = error("overload vec_layout(::$(typeof(lay)), _)")


"""
    vec(a::AbstractQuasiMatrix)

reshapes a quasi-matrix into quasi-vector. The axes of the resulting
quasi-vector depends on the axes of the quasi-matrix: if the axes are
`(ax, Base.OneTo(1))` or `(Base.OneTo(1), ax)` then the unary dimension is dropped.
If the axes are both continuous then it may form a quasi-vector defined on the rectangle,
if MultivariateOrthogonalPolynomials.jl is loaded.
"""
vec(a::AbstractQuasiMatrix) = vec_layout(MemoryLayout(a), a)


reshape(parent::AbstractQuasiArray{T,N}, ndims::Val{N}) where {T,N} = parent
function reshape(parent::AbstractQuasiArray, ndims::Val{N}) where N
    reshape(parent, rdims(Val(N), axes(parent)))
end


reshape_layout(lay, a, dims) = error("overload reshape_layout(::$(typeof(lay)), _, $(typeof(dims)))")


"""
   reshape(a::AbstractQuasiVector)

will reshape a quasi-vector defined on a rectangle to a quasi-matrix.
"""
reshape(a::AbstractQuasiArray, dims...) = reshape_layout(MemoryLayout(a), a, dims...)