module QuasiArraysSparseArraysExt

using QuasiArrays, SparseArrays

SparseArrays.issparse(::AbstractQuasiArray) = false

end
