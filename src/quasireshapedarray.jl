reshape(parent::AbstractQuasiArray{T,N}, ndims::Val{N}) where {T,N} = parent
function reshape(parent::AbstractQuasiArray, ndims::Val{N}) where N
    reshape(parent, rdims(Val(N), axes(parent)))
end
