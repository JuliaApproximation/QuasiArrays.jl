# This file is based on a part of Julia. License is MIT: https://julialang.org/license

_mapreduce(f, op, ::IndexCartesian, A::AbstractQuasiArray) = mapfoldl(f, op, A)

## Functions to compute the reduced shape

# This differs from Arrays in that we always have Base.OneTo
reduced_index(i::Inclusion) = Base.OneTo(1)
reduced_indices(a::AbstractQuasiArray, region) = reduced_indices(axes(a), region)

# for reductions that keep 0 dims as 0
reduced_indices0(a::AbstractQuasiArray, region) = reduced_indices0(axes(a), region)

function reduced_indices(inds::NTuple{N,Any}, d::Int) where N
    d < 1 && throw(ArgumentError("dimension must be ≥ 1, got $d"))
    if d == 1
        return (reduced_index(inds[1]), tail(inds)...)
    elseif 1 < d <= N
        return tuple(inds[1:d-1]..., reduced_index(inds[d]), inds[d+1:N]...)
    else
        return inds
    end
end

function reduced_indices0(inds::NTuple{N,Any}, d::Int) where N
    d < 1 && throw(ArgumentError("dimension must be ≥ 1, got $d"))
    if d <= N
        ind = inds[d]
        rd = isempty(ind) ? ind : reduced_index(inds[d])
        if d == 1
            return (rd, tail(inds)...)::typeof(inds)
        else
            return tuple(inds[1:d-1]..., oftype(inds[d], rd), inds[d+1:N]...)::typeof(inds)
        end
    else
        return inds
    end
end

function reduced_indices(inds::NTuple{N,Any}, region) where N
    rinds = [inds...]
    for i in region
        isa(i, Integer) || throw(ArgumentError("reduced dimension(s) must be integers"))
        d = Int(i)
        if d < 1
            throw(ArgumentError("region dimension(s) must be ≥ 1, got $d"))
        elseif d <= N
            rinds[d] = reduced_index(rinds[d])
        end
    end
    tuple(rinds...)::typeof(inds)
end

function reduced_indices0(inds::NTuple{N,Any}, region) where N
    rinds = [inds...]
    for i in region
        isa(i, Integer) || throw(ArgumentError("reduced dimension(s) must be integers"))
        d = Int(i)
        if d < 1
            throw(ArgumentError("region dimension(s) must be ≥ 1, got $d"))
        elseif d <= N
            rind = rinds[d]
            rinds[d] = isempty(rind) ? rind : reduced_index(rind)
        end
    end
    tuple(rinds...)::typeof(inds)
end

###### Generic reduction functions #####

## initialization
# initarray! is only called by sum!, prod!, etc.
for (Op, initfun) in ((:(typeof(add_sum)), :zero), (:(typeof(mul_prod)), :one))
    @eval initarray!(a::AbstractQuasiArray{T}, ::$(Op), init::Bool, src::AbstractQuasiArray) where {T} = (init && fill!(a, $(initfun)(T)); a)
end

for Op in (:(typeof(max)), :(typeof(min)))
    @eval initarray!(a::AbstractQuasiArray{T}, ::$(Op), init::Bool, src::AbstractQuasiArray) where {T} = (init && copyfirst!(a, src); a)
end

for (Op, initval) in ((:(typeof(&)), true), (:(typeof(|)), false))
    @eval initarray!(a::AbstractQuasiArray, ::$(Op), init::Bool, src::AbstractQuasiArray) = (init && fill!(a, $initval); a)
end

# reducedim_initarray is called by
reducedim_initarray(A::AbstractQuasiArray, region, init, ::Type{R}) where {R} = fill!(similar(A,R,reduced_indices(A,region)), init)
reducedim_initarray(A::AbstractQuasiArray, region, init::T) where {T} = reducedim_initarray(A, region, init, T)


function reducedim_init(f, op::Union{typeof(+),typeof(add_sum)}, A::AbstractQuasiArray, region)
    _reducedim_init(f, op, zero, sum, A, region)
end
function reducedim_init(f, op::Union{typeof(*),typeof(mul_prod)}, A::AbstractQuasiArray, region)
    _reducedim_init(f, op, one, prod, A, region)
end

# initialization when computing minima and maxima requires a little care
for (f1, f2, initval, typeextreme) in ((:min, :max, :Inf, :typemax), (:max, :min, :(-Inf), :typemin))
    @eval function reducedim_init(f, op::typeof($f1), A::AbstractQuasiArray, region)
        # First compute the reduce indices. This will throw an ArgumentError
        # if any region is invalid
        ri = reduced_indices(A, region)

        # Next, throw if reduction is over a region with length zero
        any(i -> isempty(axes(A, i)), region) && _empty_reduce_error()

        # Make a view of the first slice of the region
        A1 = view(A, ri...)

        if isempty(A1)
            # If the slice is empty just return non-view version as the initial array
            return copy(A1)
        else
            # otherwise use the min/max of the first slice as initial value
            v0 = mapreduce(f, $f2, A1)

            T = _realtype(f, promote_union(eltype(A)))
            Tr = v0 isa T ? T : typeof(v0)

            # but NaNs and missing need to be avoided as initial values
            if (v0 == v0) === false
                # v0 is NaN
                v0 = $initval
            elseif isunordered(v0)
                # v0 is missing or a third-party unordered value
                Tnm = nonmissingtype(Tr)
                # TODO: Some types, like BigInt, don't support typemin/typemax.
                # So a Matrix{Union{BigInt, Missing}} can still error here.
                v0 = $typeextreme(Tnm)
            end
            # v0 may have changed type.
            Tr = v0 isa T ? T : typeof(v0)

            return reducedim_initarray(A, region, v0, Tr)
        end
    end
end
reducedim_init(f::Union{typeof(abs),typeof(abs2)}, op::typeof(max), A::AbstractQuasiArray{T}, region) where {T} =
    reducedim_initarray(A, region, zero(f(zero(T))), _realtype(f, T))

reducedim_init(f, op::typeof(&), A::AbstractQuasiArray, region) = reducedim_initarray(A, region, true)
reducedim_init(f, op::typeof(|), A::AbstractQuasiArray, region) = reducedim_initarray(A, region, false)

# specialize to make initialization more efficient for common cases

let
    BitIntFloat = Union{BitInteger, IEEEFloat}
    T = Union{
        [AbstractQuasiArray{t} for t in uniontypes(BitIntFloat)]...,
        [AbstractQuasiArray{Complex{t}} for t in uniontypes(BitIntFloat)]...}

    global function reducedim_init(f, op::Union{typeof(+),typeof(add_sum)}, A::T, region)
        z = zero(f(zero(eltype(A))))
        reducedim_initarray(A, region, op(z, z))
    end
    global function reducedim_init(f, op::Union{typeof(*),typeof(mul_prod)}, A::T, region)
        u = one(f(one(eltype(A))))
        reducedim_initarray(A, region, op(u, u))
    end
end

## generic (map)reduction

has_fast_linear_indexing(a::AbstractQuasiArray) = false


copyfirst!(R::AbstractQuasiArray, A::AbstractQuasiArray) = mapfirst!(identity, R, A)

function mapfirst!(f::F, R::AbstractQuasiArray, A::AbstractQuasiArray{<:Any,N}) where {N, F}
    lsiz = check_reducedims(R, A)
    t = _firstreducedslice(axes(R), axes(A))
    map!(f, R, view(A, t...))
end

function _mapreducedim!(f, op, R::AbstractQuasiArray, A::AbstractQuasiArray)
    lsiz = check_reducedims(R,A)
    isempty(A) && return R

    if has_fast_linear_indexing(A) && lsiz > 16
        # use mapreduce_impl, which is probably better tuned to achieve higher performance
        nslices = div(length(A), lsiz)
        ibase = first(LinearIndices(A))-1
        for i = 1:nslices
            @inbounds R[i] = op(R[i], mapreduce_impl(f, op, A, ibase+1, ibase+lsiz))
            ibase += lsiz
        end
        return R
    end
    indsAt, indsRt = safe_tail(axes(A)), safe_tail(axes(R)) # handle d=1 manually
    keep, Idefault = Broadcast.shapeindexer(indsRt)
    if reducedim1(R, A)
        # keep the accumulator as a local variable when reducing along the first dimension
        i1 = first(axes1(R))
        @inbounds for IA in QuasiCartesianIndices(indsAt)
            IR = Broadcast.newindex(IA, keep, Idefault)
            r = R[i1,IR]
            for i in axes(A, 1)
                r = op(r, f(A[i, IA]))
            end
            R[i1,IR] = r
        end
    else
        @inbounds for IA in QuasiCartesianIndices(indsAt)
            IR = Broadcast.newindex(IA, keep, Idefault)
            for i in axes(A, 1)
                R[i,IR] = op(R[i,IR], f(A[i,IA]))
            end
        end
    end
    return R
end

mapreducedim!(f, op, R::AbstractQuasiArray, A::AbstractQuasiArray) =
    (_mapreducedim!(f, op, R, A); R)

reducedim!(op, R::AbstractQuasiArray{RT}, A::AbstractQuasiArray) where {RT} =
    mapreducedim!(identity, op, R, A)

mapreduce(f, op, A::AbstractQuasiArray; dims=:, init=_InitialValue()) =
    _mapreduce_dim(f, op, init, A, dims)
mapreduce(f, op, A::AbstractQuasiArray...; kw...) =
    reduce(op, map(f, A...); kw...)

_mapreduce_dim(f, op, nt, A::AbstractQuasiArray, ::Colon) =
    mapfoldl_impl(f, op, nt, A)

_mapreduce_dim(f, op, ::_InitialValue, A::AbstractQuasiArray, ::Colon) =
    _mapreduce(f, op, IndexStyle(A), A)

_mapreduce_dim(f, op, nt, A::AbstractQuasiArray, dims) =
    mapreducedim!(f, op, reducedim_initarray(A, dims, nt), A)

_mapreduce_dim(f, op, ::_InitialValue, A::AbstractQuasiArray, dims) =
    mapreducedim!(f, op, reducedim_init(f, op, A, dims), A)

count(A::AbstractQuasiArray; dims=:, init=0) = count(identity, A; dims, init)
count(f, A::AbstractQuasiArray; dims=:, init=0) = _count(f, A, dims, init)

_count(f, A::AbstractQuasiArray, dims::Colon, init) = _simple_count(f, A, init)
_count(f, A::AbstractQuasiArray, dims, init) = mapreduce(_bool(f), add_sum, A; dims, init)

count!(r::AbstractQuasiArray, A::AbstractQuasiArray; init::Bool=true) = count!(identity, r, A; init=init)
count!(f, r::AbstractQuasiArray, A::AbstractQuasiArray; init::Bool=true) =
    mapreducedim!(_bool(f), add_sum, initarray!(r, add_sum, init, A), A)


for (fname, _fname, op) in [(:sum,     :_sum,     :add_sum), (:prod,    :_prod,    :mul_prod),
                            (:maximum, :_maximum, :max),     (:minimum, :_minimum, :min)]
    @eval begin
        # User-facing methods with keyword arguments
        @inline ($fname)(a::AbstractQuasiArray; dims=:, kw...) = ($_fname)(a, dims; kw...)
        @inline ($fname)(f, a::AbstractQuasiArray; dims=:, kw...) = ($_fname)(f, a, dims; kw...)
    end
end



any(a::AbstractQuasiArray; dims=:)              = _any(a, dims)
any(f::Function, a::AbstractQuasiArray; dims=:) = _any(f, a, dims)
all(a::AbstractQuasiArray; dims=:)              = _all(a, dims)
all(f::Function, a::AbstractQuasiArray; dims=:) = _all(f, a, dims)

for (fname, op) in [(:sum, :add_sum), (:prod, :mul_prod),
                    (:maximum, :max), (:minimum, :min),
                    (:all, :&),       (:any, :|)]
    fname! = Symbol(fname, '!')
    _fname = Symbol('_', fname)
    @eval begin
        $(fname!)(f::Function, r::AbstractQuasiArray, A::AbstractQuasiArray; init::Bool=true) =
            mapreducedim!(f, $(op), initarray!(r, $(op), init, A), A)
        $(fname!)(r::AbstractQuasiArray, A::AbstractQuasiArray; init::Bool=true) = $(fname!)(identity, r, A; init=init)
    end
end

##### findmin & findmax #####
# The initial values of Rval are not used if the corresponding indices in Rind are 0.
#
function findminmax!(f, Rval, Rind, A::AbstractQuasiArray{T,N}) where {T,N}
    (isempty(Rval) || isempty(A)) && return Rval, Rind
    lsiz = check_reducedims(Rval, A)
    for i = 1:N
        axes(Rval, i) == axes(Rind, i) || throw(DimensionMismatch("Find-reduction: outputs must have the same indices"))
    end
    # If we're reducing along dimension 1, for efficiency we can make use of a temporary.
    # Otherwise, keep the result in Rval/Rind so that we traverse A in storage order.
    indsAt, indsRt = safe_tail(axes(A)), safe_tail(axes(Rval))
    keep, Idefault = Broadcast.shapeindexer(indsRt)
    ks = keys(A)
    y = iterate(ks)
    zi = zero(eltype(ks))
    if reducedim1(Rval, A)
        i1 = first(axes1(Rval))
        @inbounds for IA in CartesianIndices(indsAt)
            IR = Broadcast.newindex(IA, keep, Idefault)
            tmpRv = Rval[i1,IR]
            tmpRi = Rind[i1,IR]
            for i in axes(A,1)
                k, kss = y::Tuple
                tmpAv = A[i,IA]
                if tmpRi == zi || f(tmpRv, tmpAv)
                    tmpRv = tmpAv
                    tmpRi = k
                end
                y = iterate(ks, kss)
            end
            Rval[i1,IR] = tmpRv
            Rind[i1,IR] = tmpRi
        end
    else
        @inbounds for IA in CartesianIndices(indsAt)
            IR = Broadcast.newindex(IA, keep, Idefault)
            for i in axes(A, 1)
                k, kss = y::Tuple
                tmpAv = A[i,IA]
                tmpRv = Rval[i,IR]
                tmpRi = Rind[i,IR]
                if tmpRi == zi || f(tmpRv, tmpAv)
                    Rval[i,IR] = tmpAv
                    Rind[i,IR] = k
                end
                y = iterate(ks, kss)
            end
        end
    end
    Rval, Rind
end

function findmin!(rval::AbstractQuasiArray, rind::AbstractQuasiArray, A::AbstractQuasiArray;
                  init::Bool=true)
    findminmax!(isgreater, init && !isempty(A) ? fill!(rval, first(A)) : rval, fill!(rind,zero(eltype(keys(A)))), A)
end

findmin(A::AbstractQuasiArray; dims=:) = _findmin(A, dims)

function findmax!(rval::AbstractQuasiArray, rind::AbstractQuasiArray, A::AbstractQuasiArray;
                  init::Bool=true)
    findminmax!(isless, init && !isempty(A) ? fill!(rval, first(A)) : rval, fill!(rind,zero(eltype(keys(A)))), A)
end
findmax(A::AbstractQuasiArray; dims=:) = _findmax(A, dims)
argmin(A::AbstractQuasiArray; dims=:) = findmin(A; dims=dims)[2]
argmax(A::AbstractQuasiArray; dims=:) = findmax(A; dims=dims)[2]


# support overloading sum by MemoryLayout
_sum(V::AbstractQuasiArray, dims) = __sum(MemoryLayout(typeof(V)), V, dims)
_sum(V::AbstractQuasiArray, ::Colon) = __sum(MemoryLayout(typeof(V)), V, :)

# sum is equivalent to hitting by ones(n) on the left or rifght
function __sum(LAY::ApplyLayout{typeof(*)}, V::AbstractQuasiMatrix, d::Int)
    a = arguments(LAY, V)
    if d == 1
        *(sum(first(a); dims=1), tail(a)...)
    else
        @assert d == 2
        *(most(a)..., sum(last(a); dims=2))
    end
end

__sum(_, A, dims) = _sum(identity, A, dims)