import FillArrays: _fill_getindex, getindex_value, fillsimilar

"""
    AbstractQuasiFill{T, N, Axes} <: AbstractQuasiArray{T, N}

Supertype for lazy array types whose entries are all equal to constant.
"""
abstract type AbstractQuasiFill{T, N, Axes} <: AbstractQuasiArray{T, N} end

Base.@propagate_inbounds @inline function _fill_getindex(F::AbstractQuasiFill, kj::IND) where IND
    @boundscheck checkbounds(F, kj...)
    getindex_value(F)
end

==(a::AbstractQuasiFill, b::AbstractQuasiFill) = axes(a) == axes(b) && getindex_value(a) == getindex_value(b)

_getindex(::Type{IND}, F::AbstractQuasiFill, k::IND) where IND = _fill_getindex(F, k)

@inline function _setindex!(::Type{IND}, F::AbstractQuasiFill, v, k::IND) where IND
    @boundscheck checkbounds(F, k...)
    v == getindex_value(F) || throw(ArgumentError("Cannot setindex! to $v for an AbstractQuasiFill with value $(getindex_value(F))."))
    F
end

@inline function fill!(F::AbstractQuasiFill, v)
    v == getindex_value(F) || throw(ArgumentError("Cannot fill! with $v an AbstractQuasiFill with value $(getindex_value(F))."))
    F
end

rank(F::AbstractQuasiFill) = iszero(getindex_value(F)) ? 0 : 1


"""
    QuasiFill{T, N, Axes}

A lazy representation of a quasi-array of dimension `N`
whose entries are all equal to a constant of type `T`,
with axes of type `Axes`.
Typically created by `QuasiFill`
```
"""
struct QuasiFill{T, N, Axes} <: AbstractQuasiFill{T, N, Axes}
    value::T
    axes::Axes

    QuasiFill{T,N,Axes}(x::T, sz::Axes) where Axes<:Tuple{Vararg{Any,N}} where {T, N} =
        new{T,N,Axes}(x,sz)
end

QuasiFill{T,N,Axes}(x, sz::Axes) where Axes<:Tuple{Vararg{Any,N}} where {T, N} =
    QuasiFill{T,N,Axes}(convert(T, x)::T, sz)

@inline QuasiFill{T, N}(x::T, sz::Axes) where Axes<:Tuple{Vararg{Any,N}} where {T, N} =
    QuasiFill{T,N,Axes}(x, sz)
@inline QuasiFill{T, N}(x, sz::Axes) where Axes<:Tuple{Vararg{Any,N}} where {T, N} =
    QuasiFill{T,N}(convert(T, x)::T, sz)

QuasiFill{T,N}(x, sz...) where {T,N} = QuasiFill{T,N}(x, map(_inclusion, sz))
QuasiFill{T}(x, sz...) where T = QuasiFill{T}(x, map(_inclusion, sz))
QuasiFill(x, sz...) = QuasiFill(x, map(_inclusion, sz))

# We restrict to  when T is specified to avoid ambiguity with a QuasiFill of a QuasiFill
@inline QuasiFill{T}(F::QuasiFill{T}) where T = F
@inline QuasiFill{T,N}(F::QuasiFill{T,N}) where {T,N} = F
@inline QuasiFill{T,N,Axes}(F::QuasiFill{T,N,Axes}) where {T,N,Axes} = F

@inline axes(F::QuasiFill) = F.axes
@inline size(F::QuasiFill) = length.(F.axes)

@inline getindex_value(F::QuasiFill) = F.value

AbstractQuasiArray{T}(F::QuasiFill{T}) where T = F
AbstractQuasiArray{T,N}(F::QuasiFill{T,N}) where {T,N} = F
AbstractQuasiArray{T}(F::QuasiFill{V,N}) where {T,V,N} = QuasiFill{T}(convert(T, F.value)::T, F.axes)
AbstractQuasiArray{T,N}(F::QuasiFill{V,N}) where {T,V,N} = QuasiFill{T}(convert(T, F.value)::T, F.axes)

convert(::Type{AbstractQuasiArray{T}}, F::QuasiFill{T}) where T = F
convert(::Type{AbstractQuasiArray{T,N}}, F::QuasiFill{T,N}) where {T,N} = F
convert(::Type{AbstractQuasiArray{T}}, F::QuasiFill) where {T} = AbstractQuasiArray{T}(F)
convert(::Type{AbstractQuasiArray{T,N}}, F::QuasiFill) where {T,N} = AbstractQuasiArray{T,N}(F)
convert(::Type{AbstractQuasiFill}, F::AbstractQuasiFill) = F
convert(::Type{AbstractQuasiFill{T}}, F::AbstractQuasiFill) where T = convert(AbstractQuasiArray{T}, F)
convert(::Type{AbstractQuasiFill{T,N}}, F::AbstractQuasiFill) where {T,N} = convert(AbstractQuasiArray{T,N}, F)

copy(F::QuasiFill) = QuasiFill(F.value, F.axes)

# ambiguity fix
convert(::Type{T}, F::T) where T<:QuasiFill = F



getindex(F::QuasiFill{<:Any,0}) = getindex_value(F)


sort(a::AbstractQuasiFill; kwds...) = a
sort!(a::AbstractQuasiFill; kwds...) = a

+(a::AbstractQuasiFill) = a
-(a::AbstractQuasiFill) = QuasiFill(-getindex_value(a), axes(a))

# Fill +/- Fill
function +(a::AbstractQuasiFill{T, N}, b::AbstractQuasiFill{V, N}) where {T, V, N}
    axes(a) ≠ axes(b) && throw(DimensionMismatch("dimensions must match."))
    return QuasiFill(getindex_value(a) + getindex_value(b), axes(a))
end
-(a::AbstractQuasiFill, b::AbstractQuasiFill) = a + (-b)

for (Typ, funcs, func) in ((:QuasiZeros, :zeros, :zero), (:QuasiOnes, :ones, :one))
    @eval begin
        """ `$($Typ){T, N, Axes} <: AbstractQuasiFill{T, N, Axes}` (lazy `$($funcs)` with axes)"""
        struct $Typ{T, N, Axes} <: AbstractQuasiFill{T, N, Axes}
            axes::Axes
            @inline $Typ{T,N,Axes}(sz::Axes) where Axes<:Tuple{Vararg{Any,N}} where {T, N} =
                new{T,N,Axes}(sz)
            @inline $Typ{T,N}(sz::Axes) where Axes<:Tuple{Vararg{Any,N}} where {T, N} =
                new{T,N,Axes}(sz)
        end
        @inline $Typ{T}(sz::SZ) where SZ<:Tuple{Vararg{Any,N}} where {T, N} = $Typ{T, N}(sz)
        @inline $Typ(sz::SZ) where SZ<:Tuple{Vararg{Any,N}} where N = $Typ{Float64,N}(sz)
        @inline $Typ(sz...) = $Typ(map(_inclusion,sz))
        @inline $Typ{T}(sz...) where T = $Typ{T}(map(_inclusion,sz))

        @inline axes(Z::$Typ) = Z.axes
        @inline size(Z::$Typ) = length.(Z.axes)
        @inline getindex_value(Z::$Typ{T}) where T = $func(T)

        AbstractQuasiArray{T}(F::$Typ{T}) where T = F
        AbstractQuasiArray{T,N}(F::$Typ{T,N}) where {T,N} = F
        AbstractQuasiArray{T}(F::$Typ) where T = $Typ{T}(F.axes)
        AbstractQuasiArray{T,N}(F::$Typ{V,N}) where {T,V,N} = $Typ{T}(F.axes)
        convert(::Type{AbstractQuasiArray{T}}, F::$Typ{T}) where T = AbstractQuasiArray{T}(F)
        convert(::Type{AbstractQuasiArray{T,N}}, F::$Typ{T,N}) where {T,N} = AbstractQuasiArray{T,N}(F)
        convert(::Type{AbstractQuasiArray{T}}, F::$Typ) where T = AbstractQuasiArray{T}(F)
        convert(::Type{AbstractQuasiArray{T,N}}, F::$Typ) where {T,N} = AbstractQuasiArray{T,N}(F)

        copy(F::$Typ) = F

        getindex(F::$Typ{T,0}) where T = getindex_value(F)
    end
end


"""
    fillsimilar(a::AbstractQuasiFill, axes)

creates a fill object that has the same fill value as `a` but
with the specified axes.
For example, if `a isa QuasiZeros` then so is the returned object.
"""
fillsimilar(a::QuasiOnes{T}, axes...) where T = QuasiOnes{T}(axes)
fillsimilar(a::QuasiZeros{T}, axes...) where T = QuasiZeros{T}(axes)
fillsimilar(a::AbstractQuasiFill, axes...) = QuasiFill(getindex_value(a), axes)


rank(F::QuasiZeros) = 0
rank(F::QuasiOnes) = 1

const QuasiEye{T,Axes} = QuasiDiagonal{T,QuasiOnes{T,1,Tuple{Axes}}}

@inline QuasiEye{T}(n::Integer) where T = Diagonal(Ones{T}(n))
@inline QuasiEye(n::Integer) = Diagonal(Ones(n))
@inline QuasiEye{T}(ax::Tuple{AbstractUnitRange{Int}}) where T = Diagonal(Ones{T}(ax))
@inline QuasiEye(ax::Tuple{AbstractUnitRange{Int}}) = Diagonal(Ones(ax))


# function iterate(iter::Eye, istate = (1, 1))
#     (i::Int, j::Int) = istate
#     m = size(iter, 1)
#     return i > m ? nothing :
#         ((@inbounds getindex(iter, i, j)),
#          j == m ? (i + 1, 1) : (i, j + 1))
# end

isone(::QuasiEye) = true

@inline QuasiEye{T}(A::AbstractQuasiMatrix) where T = QuasiEye{T}(axes(A))
@inline QuasiEye(A::AbstractQuasiMatrix) = QuasiEye{eltype(A)}(axes(A))


#########
#  Special matrix types
#########


function convert(::Type{QuasiDiagonal}, Z::QuasiZeros{T,2}) where T
    n,m = axes(Z)
    n ≠ m && throw(BoundsError(Z))
    QuasiDiagonal(QuasiZeros{T}((n,)))
end

function convert(::Type{QuasiDiagonal{T}}, Z::QuasiZeros{V,2}) where {T,V}
    n,m = axes(Z)
    n ≠ m && throw(BoundsError(Z))
    QuasiDiagonal(QuasiZeros{T}((n,)))
end


#########
# maximum/minimum
#########

for op in (:maximum, :minimum)
    @eval $op(x::AbstractQuasiFill) = getindex_value(x)
end


#########
# Cumsum
#########

sum(x::AbstractQuasiFill) = getindex_value(x)*measure(axes(x,1))
sum(x::QuasiZeros) = getindex_value(x)

# define `sum(::Callable, ::AbstractQuasiFill)` to avoid method ambiguity errors on Julia 1.0
sum(f, x::AbstractQuasiFill) = _sum(f, x)
sum(f::Base.Callable, x::AbstractQuasiFill) = _sum(f, x)
_sum(f, x::AbstractQuasiFill) = measure(x) * f(getindex_value(x))


#########
# zero
#########

zero(r::QuasiZeros{T,N}) where {T,N} = r
zero(r::QuasiOnes{T,N}) where {T,N} = QuasiZeros{T,N}(r.axes)
zero(r::QuasiFill{T,N}) where {T,N} = QuasiZeros{T,N}(r.axes)


#########
# in
#########
in(x, A::AbstractQuasiFill) = x == getindex_value(A)
