module QuasiArraysRecipesBaseExt

using RecipesBase, QuasiArrays

@recipe f(v::Inclusion{<:Any,<:AbstractVector}) = (v.domain,)
@recipe f(v::QuasiVector) = v.axes[1], v.parent

end # module