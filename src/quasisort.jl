for func in (:findall, :findfirst, :findlast)
    func_layout = Symbol(string(func) * "_layout")
    @eval begin
       $func(f::Function, v::AbstractQuasiVector; kwds...) = $func_layout(MemoryLayout(v), f, v; kwds...) 
        function $func_layout(lay, f, v::QuasiVector; kwds...)
            inds = $func(f, parent(v); kwds...)
            inds isa Nothing && return nothing
            v.axes[1][inds]
        end
    end
end

for func in (:searchsortedfirst, :searchsorted, :searchsortedlast)
    func_layout = Symbol(string(func) * "_layout")
    @eval begin
       $func(f::AbstractQuasiVector, x; kwds...) = $func_layout(MemoryLayout(f), f, x; kwds...) 
        function $func_layout(lay, f::QuasiVector, x; kwds...)
            inds = $func(parent(f), x; kwds...)
            f.axes[1][inds]
        end
    end
end

