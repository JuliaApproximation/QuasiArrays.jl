import FillArrays: getindex_value, fillsimilar, fillzero

"""
    AbstractQuasiFill{T, N, Axes} <: AbstractQuasiArray{T, N}

Supertype for lazy array types whose entries are all equal to constant.
"""
abstract type AbstractQuasiFill{T, N, Axes} <: AbstractQuasiArray{T, N} end

Base.@propagate_inbounds @inline function _fill_getindex(::Type{IND}, F::AbstractQuasiFill, kj::IND) where IND
    @boundscheck checkbounds(F, kj...)
    getindex_value(F)
end

Base.@propagate_inbounds @inline function _fill_getindex(::Type{IND}, A::AbstractQuasiFill{<:Any,N}, I) where {N,IND}
    @boundscheck checkbounds(A, I...)
    shape = Base.index_shape(I...)
    fillsimilar(A, shape...)
end

==(a::AbstractQuasiFill, b::AbstractQuasiFill) = axes(a) == axes(b) && getindex_value(a) == getindex_value(b)

_getindex(::Type{IND}, F::AbstractQuasiFill, k::IND) where IND = _fill_getindex(IND, F, k)


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
QuasiFill{T}(x, sz::NTuple{N,Any}) where {T,N} = QuasiFill{T,N}(x, sz)
QuasiFill(x::T, sz::NTuple{N,Any}) where {T,N} = QuasiFill{T,N}(x, sz)

QuasiFill{T,N}(x, sz...) where {T,N} = QuasiFill{T,N}(x, map(_inclusion, sz))
QuasiFill{T}(x, sz...) where T = QuasiFill{T}(x, map(_inclusion, sz))
QuasiFill(x, sz...) = QuasiFill(x, map(_inclusion, sz))

# We restrict to  when T is specified to avoid ambiguity with a QuasiFill of a QuasiFill
@inline QuasiFill{T}(F::QuasiFill{T}) where T = F
@inline QuasiFill{T,N}(F::QuasiFill{T,N}) where {T,N} = F
@inline QuasiFill{T,N,Axes}(F::QuasiFill{T,N,Axes}) where {T,N,Axes} = F

@inline axes(F::QuasiFill) = F.axes
@inline size(F::QuasiFill) = length.(F.axes)
MemoryLayout(::Type{<:QuasiFill}) = FillLayout()

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

fillsimilar(a::QuasiOnes{T}, axes::AbstractUnitRange{Int}...) where T = Ones{T}(axes)
fillsimilar(a::QuasiZeros{T}, axes::AbstractUnitRange{Int}...) where T = Zeros{T}(axes)
fillsimilar(a::AbstractQuasiFill, axes::AbstractUnitRange{Int}...) = Fill(getindex_value(a), axes)



rank(F::QuasiZeros) = 0
rank(F::QuasiOnes) = 1

diag(F::QuasiZeros{T,2}) where T = QuasiZeros{T}((axes(F,1),))

const QuasiEye{T,Axes} = QuasiDiagonal{T,QuasiOnes{T,1,Tuple{Axes}}}

@inline QuasiEye{T}(n) where T = QuasiDiagonal(QuasiOnes{T}(n))
@inline QuasiEye(n) = QuasiDiagonal(QuasiOnes(n))

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

quasiscaling(J, ax) = QuasiDiagonal(QuasiFill(J.λ, ax))

for op in (:+, :-, :*, :/, :\)
    @eval begin
        $op(A::AbstractQuasiMatrix, J::UniformScaling) = $op(A, quasiscaling(J, axes(A,2)))
        $op(J::UniformScaling, A::AbstractQuasiMatrix) = $op(quasiscaling(J, axes(A,1)), A)
    end
end


########
# show
#######

function summary(io::IO, F::QuasiOnes)
    print(io, "ones(")
    summary(io, F.axes[1])
    for a in tail(F.axes)
        print(io, ", ")
        summary(io, a)
    end
    print(io, ")")
end

function summary(io::IO, F::QuasiZeros)
    print(io, "zeros(")
    summary(io, F.axes[1])
    for a in tail(F.axes)
        print(io, ", ")
        summary(io, a)
    end
    print(io, ")")
end

function summary(io::IO, F::QuasiFill)
    print(io, "fill($(F.value), ")
    summary(io, F.axes[1])
    for a in tail(F.axes)
        print(io, ", ")
        summary(io, a)
    end
    print(io, ")")
end

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



####
# Algebra
####

+(a::AbstractQuasiFill) = a
-(a::AbstractQuasiFill) = QuasiFill(-getindex_value(a), axes(a))

# Fill +/- Fill
function +(a::AbstractQuasiFill{T, N}, b::AbstractQuasiFill{V, N}) where {T, V, N}
    axes(a) ≠ axes(b) && throw(DimensionMismatch("dimensions must match."))
    return QuasiFill(getindex_value(a) + getindex_value(b), axes(a))
end
-(a::AbstractQuasiFill, b::AbstractQuasiFill) = a + (-b)


## Transpose/Adjoint
# cannot do this for vectors since that would destroy scalar dot product


transpose(a::QuasiOnes{T,2}) where T = QuasiOnes{T}(reverse(a.axes))
adjoint(a::QuasiOnes{T,2}) where T = QuasiOnes{T}(reverse(a.axes))
transpose(a::QuasiZeros{T,2}) where T = QuasiZeros{T}(reverse(a.axes))
adjoint(a::QuasiZeros{T,2}) where T = QuasiZeros{T}(reverse(a.axes))
transpose(a::QuasiFill{T,2}) where T = QuasiFill{T}(transpose(a.value), reverse(a.axes))
adjoint(a::QuasiFill{T,2}) where T = QuasiFill{T}(adjoint(a.value), reverse(a.axes))

permutedims(a::AbstractQuasiFill{<:Any,1}) = fillsimilar(a, Inclusion(1), axes(a,1))
permutedims(a::AbstractQuasiFill{<:Any,2}) = fillsimilar(a, reverse(a.axes)...)


## Algebraic identities

+(a::QuasiZeros) = a
-(a::QuasiZeros) = a

# Zeros +/- Zeros
function +(a::QuasiZeros{T}, b::QuasiZeros{V}) where {T, V}
    axes(a) ≠ axes(b) && throw(DimensionMismatch("dimensions must match."))
    return QuasiZeros{promote_type(T,V)}(axes(a))
end
-(a::QuasiZeros, b::QuasiZeros) = -(a + b)
-(a::QuasiOnes, b::QuasiOnes) = QuasiZeros(a)+QuasiZeros(b)

# Zeros +/- Fill and Fill +/- Zeros
function +(a::AbstractQuasiFill{T}, b::QuasiZeros{V}) where {T, V}
    axes(a) ≠ axes(b) && throw(DimensionMismatch("dimensions must match."))
    return convert(AbstractQuasiFill{promote_type(T, V)}, a)
end
+(a::QuasiZeros, b::AbstractQuasiFill) = b + a
-(a::AbstractQuasiFill, b::QuasiZeros) = a + b
-(a::QuasiZeros, b::AbstractQuasiFill) = a + (-b)

# QuasiZeros +/- Array and Array +/- QuasiZeros
function +(a::QuasiZeros{T, N}, b::AbstractQuasiArray{V, N}) where {T, V, N}
    axes(a) ≠ axes(b) && throw(DimensionMismatch("dimensions must match."))
    return AbstractQuasiArray{promote_type(T,V),N}(b)
end
function +(a::QuasiArray{T, N}, b::QuasiZeros{V, N}) where {T, V, N}
    axes(a) ≠ axes(b) && throw(DimensionMismatch("dimensions must match."))
    return AbstractQuasiArray{promote_type(T,V),N}(a)
end

function -(a::QuasiZeros{T, N}, b::AbstractQuasiArray{V, N}) where {T, V, N}
    axes(a) ≠ axes(b) && throw(DimensionMismatch("dimensions must match."))
    return -b + a
end
-(a::QuasiArray{T, N}, b::QuasiZeros{V, N}) where {T, V, N} = a + b


##
# view
##

Base.@propagate_inbounds _unsafe_view(::Type{IND}, A::AbstractQuasiFill{<:Any,N}, I::Tuple) where {IND,N} =
    _fill_getindex(IND, A, Base.to_indices(A,I))

# not getindex since we need array-like indexing
Base.@propagate_inbounds function _unsafe_view(::Type{IND}, A::AbstractQuasiFill{<:Any,N}, I::Vararg{IND, N}) where {N,IND}
    @boundscheck checkbounds(A, I...)
    fillsimilar(A)
end

####
# broadcast
####

### map

map(f::Function, r::AbstractQuasiFill) = QuasiFill(f(getindex_value(r)), axes(r))


### Unary broadcasting

function broadcasted(::AbstractQuasiArrayStyle{N}, op, r::AbstractQuasiFill{T,N}) where {T,N}
    return QuasiFill(op(getindex_value(r)), axes(r))
end

broadcasted(::AbstractQuasiArrayStyle, ::typeof(+), r::QuasiZeros) = r
broadcasted(::AbstractQuasiArrayStyle, ::typeof(-), r::QuasiZeros) = r
broadcasted(::AbstractQuasiArrayStyle, ::typeof(+), r::QuasiOnes) = r

broadcasted(::AbstractQuasiArrayStyle{N}, ::typeof(conj), r::QuasiZeros{T,N}) where {T,N} = r
broadcasted(::AbstractQuasiArrayStyle{N}, ::typeof(conj), r::QuasiOnes{T,N}) where {T,N} = r
broadcasted(::AbstractQuasiArrayStyle{N}, ::typeof(real), r::QuasiZeros{T,N}) where {T,N} = QuasiZeros{real(T)}(r.axes)
broadcasted(::AbstractQuasiArrayStyle{N}, ::typeof(real), r::QuasiOnes{T,N}) where {T,N} = QuasiOnes{real(T)}(r.axes)
broadcasted(::AbstractQuasiArrayStyle{N}, ::typeof(imag), r::QuasiZeros{T,N}) where {T,N} = QuasiZeros{real(T)}(r.axes)
broadcasted(::AbstractQuasiArrayStyle{N}, ::typeof(imag), r::QuasiOnes{T,N}) where {T,N} = QuasiZeros{real(T)}(r.axes)

### Binary broadcasting

function broadcasted(::AbstractQuasiArrayStyle, op, a::AbstractQuasiFill, b::AbstractQuasiFill)
    val = op(getindex_value(a), getindex_value(b))
    return QuasiFill(val, broadcast_shape(axes(a), axes(b)))
end


_broadcasted_zeros(f, a, b) = QuasiZeros{Base.Broadcast.combine_eltypes(f, (a, b))}(broadcast_shape(axes(a), axes(b)))
_broadcasted_ones(f, a, b) = QuasiOnes{Base.Broadcast.combine_eltypes(f, (a, b))}(broadcast_shape(axes(a), axes(b)))
_broadcasted_nan(f, a, b) = QuasiFill(convert(Base.Broadcast.combine_eltypes(f, (a, b)), NaN), broadcast_shape(axes(a), axes(b)))


broadcasted(::AbstractQuasiArrayStyle, ::typeof(+), a::QuasiZeros, b::QuasiZeros) = _broadcasted_zeros(+, a, b)
broadcasted(::AbstractQuasiArrayStyle, ::typeof(+), a::QuasiOnes, b::QuasiZeros) = _broadcasted_ones(+, a, b)
broadcasted(::AbstractQuasiArrayStyle, ::typeof(+), a::QuasiZeros, b::QuasiOnes) = _broadcasted_ones(+, a, b)

broadcasted(::AbstractQuasiArrayStyle, ::typeof(-), a::QuasiZeros, b::QuasiZeros) = _broadcasted_zeros(-, a, b)
broadcasted(::AbstractQuasiArrayStyle, ::typeof(-), a::QuasiOnes, b::QuasiZeros) = _broadcasted_ones(-, a, b)
broadcasted(::AbstractQuasiArrayStyle, ::typeof(-), a::QuasiOnes, b::QuasiOnes) = _broadcasted_zeros(-, a, b)

broadcasted(::AbstractQuasiArrayStyle{1}, ::typeof(+), a::QuasiZeros{<:Any,1}, b::QuasiZeros{<:Any,1}) = _broadcasted_zeros(+, a, b)
broadcasted(::AbstractQuasiArrayStyle{1}, ::typeof(+), a::QuasiOnes{<:Any,1}, b::QuasiZeros{<:Any,1}) = _broadcasted_ones(+, a, b)
broadcasted(::AbstractQuasiArrayStyle{1}, ::typeof(+), a::QuasiZeros{<:Any,1}, b::QuasiOnes{<:Any,1}) = _broadcasted_ones(+, a, b)

broadcasted(::AbstractQuasiArrayStyle{1}, ::typeof(-), a::QuasiZeros{<:Any,1}, b::QuasiZeros{<:Any,1}) = _broadcasted_zeros(-, a, b)
broadcasted(::AbstractQuasiArrayStyle{1}, ::typeof(-), a::QuasiOnes{<:Any,1}, b::QuasiZeros{<:Any,1}) = _broadcasted_ones(-, a, b)


broadcasted(::AbstractQuasiArrayStyle, ::typeof(*), a::QuasiZeros, b::QuasiZeros) = _broadcasted_zeros(*, a, b)

# In following, need to restrict to <: Number as otherwise we cannot infer zero from type
# TODO: generalise to things like SVector
for op in (:*, :/)
    @eval begin
        broadcasted(::AbstractQuasiArrayStyle, ::typeof($op), a::QuasiZeros, b::QuasiOnes) = _broadcasted_zeros($op, a, b)
        broadcasted(::AbstractQuasiArrayStyle, ::typeof($op), a::QuasiZeros, b::QuasiFill{<:Number}) = _broadcasted_zeros($op, a, b)
        broadcasted(::AbstractQuasiArrayStyle, ::typeof($op), a::QuasiZeros, b::Number) = _broadcasted_zeros($op, a, b)
        broadcasted(::AbstractQuasiArrayStyle, ::typeof($op), a::QuasiZeros, b::Base.Broadcast.Broadcasted) = _broadcasted_zeros($op, a, b)
        broadcasted(::AbstractQuasiArrayStyle, ::typeof($op), a::QuasiZeros, b::AbstractQuasiArray{<:Number}) = _broadcasted_zeros($op, a, b)
    end
end

for op in (:*, :\)
    @eval begin
        broadcasted(::AbstractQuasiArrayStyle, ::typeof($op), a::QuasiOnes, b::QuasiZeros) = _broadcasted_zeros($op, a, b)
        broadcasted(::AbstractQuasiArrayStyle, ::typeof($op), a::QuasiFill{<:Number}, b::QuasiZeros) = _broadcasted_zeros($op, a, b)
        broadcasted(::AbstractQuasiArrayStyle, ::typeof($op), a::Number, b::QuasiZeros) = _broadcasted_zeros($op, a, b)
        broadcasted(::AbstractQuasiArrayStyle, ::typeof($op), a::Base.Broadcast.Broadcasted, b::QuasiZeros) = _broadcasted_zeros($op, a, b)
        broadcasted(::AbstractQuasiArrayStyle, ::typeof($op), a::AbstractQuasiArray{<:Number}, b::QuasiZeros) = _broadcasted_zeros($op, a, b)
    end
end

for op in (:*, :/, :\)
    @eval broadcasted(::AbstractQuasiArrayStyle, ::typeof($op), a::QuasiOnes, b::QuasiOnes) = _broadcasted_ones($op, a, b)
end

for op in (:/, :\)
    @eval broadcasted(::AbstractQuasiArrayStyle, ::typeof($op), a::QuasiZeros{<:Number}, b::QuasiZeros{<:Number}) = _broadcasted_nan($op, a, b)
end


for op in (:+, -)
    @eval begin
        function broadcasted(::AbstractQuasiArrayStyle{N}, ::typeof($op), a::AbstractQuasiArray{T,N}, b::QuasiZeros{V,N}) where {T,V,N}
            broadcast_shape(axes(a), axes(b)) == axes(a) || throw(ArgumentError("Cannot broadcast $a and $b. Convert $b to a Vector first."))
            LinearAlgebra.copy_oftype(a, promote_type(T,V))
        end

        broadcasted(::AbstractQuasiArrayStyle{1}, ::typeof($op), a::AbstractQuasiFill{T,1}, b::QuasiZeros{V,1}) where {T,V} = 
            Base.invoke(broadcasted, Tuple{DefaultQuasiArrayStyle, typeof($op), AbstractQuasiFill, AbstractQuasiFill}, DefaultQuasiArrayStyle{1}(), $op, a, b)
    end
end

function broadcasted(::AbstractQuasiArrayStyle{1}, ::typeof(+), a::QuasiZeros{T,N}, b::AbstractQuasiArray{V,N}) where {T,V,N}
    broadcast_shape(axes(a), axes(b)) == axes(a) || throw(ArgumentError("Cannot broadcast $a and $b. Convert $a to a Vector first."))
    LinearAlgebra.copy_oftype(b, promote_type(T,V))
end

broadcasted(::AbstractQuasiArrayStyle{1}, ::typeof(+), a::QuasiZeros{V,1}, b::AbstractQuasiFill{T,1}) where {T,V} = 
            Base.invoke(broadcasted, Tuple{DefaultQuasiArrayStyle, typeof(+), AbstractQuasiFill, AbstractQuasiFill}, DefaultQuasiArrayStyle{1}(), +, a, b)

# Need to prevent array-valued fills from broadcasting over entry
_broadcast_getindex_value(a::AbstractQuasiFill{<:Number}) = getindex_value(a)
_broadcast_getindex_value(a::AbstractQuasiFill) = Ref(getindex_value(a))


broadcasted(::AbstractQuasiArrayStyle{N}, op, r::AbstractQuasiFill{T,N}, x::Number) where {T,N} = QuasiFill(op(getindex_value(r),x), axes(r))
broadcasted(::AbstractQuasiArrayStyle{N}, op, x::Number, r::AbstractQuasiFill{T,N}) where {T,N} = QuasiFill(op(x, getindex_value(r)), axes(r))
broadcasted(::AbstractQuasiArrayStyle{N}, op, r::AbstractQuasiFill{T,N}, x::Ref) where {T,N} = QuasiFill(op(getindex_value(r),x[]), axes(r))
broadcasted(::AbstractQuasiArrayStyle{N}, op, x::Ref, r::AbstractQuasiFill{T,N}) where {T,N} = QuasiFill(op(x[], getindex_value(r)), axes(r))

###
# Mul
###

MemoryLayout(::Type{<:QuasiZeros}) = ZerosLayout()
MemoryLayout(::Type{<:QuasiOnes}) = OnesLayout()

_quasi_mul(M::Mul{ZerosLayout}, _) = QuasiZeros{eltype(M)}(axes(M))
_quasi_mul(M::Mul{QuasiArrayLayout,ZerosLayout}, _) = QuasiZeros{eltype(M)}(axes(M))
_quasi_mul(M::Mul{QuasiArrayLayout,ZerosLayout}, ::NTuple{N,OneTo{Int}}) where N = Zeros{eltype(M)}(axes(M))
fillzeros(::Type{T}, a::Tuple{AbstractQuasiVector,Vararg{Any}}) where T<:Number = QuasiZeros{T}(a)
fillzeros(::Type{T}, a::Tuple{Any,AbstractQuasiVector,Vararg{Any}}) where T<:Number = QuasiZeros{T}(a)

copy(M::MulAdd{<:AbstractFillLayout,<:AbstractFillLayout,<:AbstractFillLayout,<:Any,<:AbstractQuasiArray}) = 
    QuasiFill(measure(axes(M.A,2))*M.α*getindex_value(M.A)*getindex_value(M.B) + M.β*getindex_value(M.C), axes(M))
copy(M::MulAdd{ZerosLayout,ZerosLayout,ZerosLayout,<:Any,<:AbstractQuasiArray}) = 
    QuasiZeros{eltype(M)}(axes(M))    
copy(M::MulAdd{ZerosLayout,<:AbstractFillLayout,ZerosLayout,<:Any,<:AbstractQuasiArray}) = 
    QuasiZeros{eltype(M)}(axes(M))
copy(M::MulAdd{<:AbstractFillLayout,ZerosLayout,ZerosLayout,<:Any,<:AbstractQuasiArray}) = 
    QuasiZeros{eltype(M)}(axes(M))


inv(D::QuasiEye) = D

zero(x::AbstractQuasiArray{T}) where {T} = QuasiZeros{T}(axes(x))
one(x::AbstractQuasiArray{T}) where {T} = QuasiOnes{T}(axes(x))

zeros(::Type{T}, x::Inclusion, y::Union{OneTo,IdentityUnitRange,Inclusion}...) where T = QuasiZeros{T}((x, y...))
zeros(x::Inclusion, y::Union{OneTo,IdentityUnitRange,Inclusion}...) = zeros(Float64, x, y...)
zeros(::Type{T}, x::Union{OneTo,IdentityUnitRange}, y::Inclusion, z::Union{OneTo,IdentityUnitRange,Inclusion}...) where T = QuasiZeros{T}((x, y, z...))
zeros(x::Union{OneTo,IdentityUnitRange}, y::Inclusion, z::Union{OneTo,IdentityUnitRange,Inclusion}...) = zeros(Float64, x, y, z...)
ones(::Type{T}, x::Inclusion, y::Union{OneTo,IdentityUnitRange,Inclusion}...) where T = QuasiOnes{T}((x, y...))
ones(x::Inclusion, y::Union{OneTo,IdentityUnitRange,Inclusion}...) = ones(Float64, x, y...)
ones(::Type{T}, x::Union{OneTo,IdentityUnitRange}, y::Inclusion, z::Union{OneTo,IdentityUnitRange,Inclusion}...) where T = QuasiOnes{T}((x, y, z...))
ones(x::Union{OneTo,IdentityUnitRange}, y::Inclusion, z::Union{OneTo,IdentityUnitRange,Inclusion}...) = ones(Float64, x, y, z...)
fill(c, x::Inclusion, y::Union{OneTo,IdentityUnitRange,Inclusion}...) where T = QuasiFill(c, (x, y...))
fill(c, x::Union{OneTo,IdentityUnitRange}, y::Inclusion, z::Union{OneTo,IdentityUnitRange,Inclusion}...) where T = QuasiFill(c, (x, y, z...))

iszero(x::AbstractQuasiFill) = iszero(getindex_value(x))
isone(x::AbstractQuasiFill) = isone(getindex_value(x))