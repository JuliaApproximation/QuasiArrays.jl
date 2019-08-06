# This file is based on a part of Julia. License is MIT: https://julialang.org/license


using QuasiArrays, Test, LinearAlgebra
@testset "QuasiAdjoint/Transpose" begin
    @testset "Adjoint and Transpose inner constructor basics" begin
        intvec, intmat = QuasiVector([1, 2],0:0.5:0.5), QuasiArray([1 2; 3 4],(0:0.5:0.5,0:0.5:0.5))
        # Adjoint/Transpose eltype must match the type of the Adjoint/Transpose of the input eltype
        @test_throws ErrorException QuasiAdjoint{Float64,typeof(intvec)}(intvec)
        @test_throws ErrorException QuasiAdjoint{Float64,typeof(intmat)}(intmat)
        @test_throws ErrorException QuasiTranspose{Float64,typeof(intvec)}(intvec)
        @test_throws ErrorException QuasiTranspose{Float64,typeof(intmat)}(intmat)
        # Adjoint/Transpose wrapped array type must match the input array type
        @test_throws MethodError QuasiAdjoint{Int,Vector{Float64}}(intvec)
        @test_throws MethodError QuasiAdjoint{Int,Matrix{Float64}}(intmat)
        @test_throws MethodError QuasiTranspose{Int,Vector{Float64}}(intvec)
        @test_throws MethodError QuasiTranspose{Int,Matrix{Float64}}(intmat)
        # Adjoint/Transpose inner constructor basic functionality, concrete scalar eltype
        @test (QuasiAdjoint{Int,typeof(intvec)}(intvec)::QuasiAdjoint{Int,typeof(intvec)}).parent === intvec
        @test (QuasiAdjoint{Int,typeof(intmat)}(intmat)::QuasiAdjoint{Int,typeof(intmat)}).parent === intmat
        @test (QuasiTranspose{Int,typeof(intvec)}(intvec)::QuasiTranspose{Int,typeof(intvec)}).parent === intvec
        @test (QuasiTranspose{Int,typeof(intmat)}(intmat)::QuasiTranspose{Int,typeof(intmat)}).parent === intmat
        # Adjoint/Transpose inner constructor basic functionality, abstract scalar eltype
        anyvec, anymat = QuasiVector(Any[1, 2],0:0.5:0.5), QuasiArray(Any[1 2; 3 4],(0:0.5:0.5,0:0.5:0.5))
        @test (QuasiAdjoint{Any,typeof(anyvec)}(anyvec)::QuasiAdjoint{Any,typeof(anyvec)}).parent === anyvec
        @test (QuasiAdjoint{Any,typeof(anymat)}(anymat)::QuasiAdjoint{Any,typeof(anymat)}).parent === anymat
        @test (QuasiTranspose{Any,typeof(anyvec)}(anyvec)::QuasiTranspose{Any,typeof(anyvec)}).parent === anyvec
        @test (QuasiTranspose{Any,typeof(anymat)}(anymat)::QuasiTranspose{Any,typeof(anymat)}).parent === anymat
    end

    @testset "Adjoint and Transpose outer constructor basics" begin
        intvec, intmat = QuasiVector([1, 2],0:0.5:0.5), QuasiArray([1 2; 3 4],(0:0.5:0.5,0:0.5:0.5))
        # the wrapped array's eltype strictly determines the Adjoint/Transpose eltype
        # so Adjoint{T}/Transpose{T} constructors are somewhat unnecessary and error-prone
        # so ascertain that such calls throw whether or not T and the input eltype are compatible
        @test_throws MethodError QuasiAdjoint{Int}(intvec)
        @test_throws MethodError QuasiAdjoint{Int}(intmat)
        @test_throws MethodError QuasiAdjoint{Float64}(intvec)
        @test_throws MethodError QuasiAdjoint{Float64}(intmat)
        @test_throws MethodError QuasiTranspose{Int}(intvec)
        @test_throws MethodError QuasiTranspose{Int}(intmat)
        @test_throws MethodError QuasiTranspose{Float64}(intvec)
        @test_throws MethodError QuasiTranspose{Float64}(intmat)
        # QuasiAdjoint/QuasiTranspose outer constructor basic functionality, concrete scalar eltype
        @test (QuasiAdjoint(intvec)::QuasiAdjoint{Int,typeof(intvec)}).parent === intvec
        @test (QuasiAdjoint(intmat)::QuasiAdjoint{Int,typeof(intmat)}).parent === intmat
        @test (QuasiTranspose(intvec)::QuasiTranspose{Int,typeof(intvec)}).parent === intvec
        @test (QuasiTranspose(intmat)::QuasiTranspose{Int,typeof(intmat)}).parent === intmat
        # the tests for the inner constructors exercise abstract scalar and concrete array eltype, forgoing here
    end

    @testset "Adjoint and Transpose add additional layers to already-wrapped objects" begin
        intvec, intmat = QuasiVector([1, 2],0:0.5:0.5), QuasiArray([1 2; 3 4],(0:0.5:0.5,0:0.5:0.5))
        @test (A = QuasiAdjoint(QuasiAdjoint(intvec))::QuasiAdjoint{Int,QuasiAdjoint{Int,typeof(intvec)}}; A.parent.parent === intvec)
        @test (A = QuasiAdjoint(QuasiAdjoint(intmat))::QuasiAdjoint{Int,QuasiAdjoint{Int,typeof(intmat)}}; A.parent.parent === intmat)
        @test (A = Transpose(Transpose(intvec))::Transpose{Int,Transpose{Int,typeof(intvec)}}; A.parent.parent === intvec)
        @test (A = Transpose(Transpose(intmat))::Transpose{Int,Transpose{Int,typeof(intmat)}}; A.parent.parent === intmat)
    end

    @testset "Adjoint and Transpose basic AbstractArray functionality" begin
        # vectors and matrices with real scalar eltype, and their adjoints/transposes
        intvec, intmat = QuasiVector([1, 2],0:0.5:0.5), QuasiMatrix([1 2 3; 4 5 6],(0:0.5:0.5, Base.OneTo(3)))
        tintvec, tintmat = QuasiMatrix([1 2],(Base.OneTo(1),0:0.5:0.5)), QuasiMatrix([1 4; 2 5; 3 6],(Base.OneTo(3),0:0.5:0.5))
        @testset "length methods" begin
            @test length(QuasiAdjoint(intvec)) == length(intvec)
            @test length(QuasiAdjoint(intmat)) == length(intmat)
            @test length(QuasiTranspose(intvec)) == length(intvec)
            @test length(Transpose(intmat)) == length(intmat)
        end
        @testset "size methods" begin
            @test size(QuasiAdjoint(intvec)) == (1, length(intvec))
            @test size(QuasiAdjoint(intmat)) == reverse(size(intmat))
            @test size(QuasiTranspose(intvec)) == (1, length(intvec))
            @test size(QuasiTranspose(intmat)) == reverse(size(intmat))
        end
        @testset "indices methods" begin
            @test axes(QuasiAdjoint(intvec)) == (Base.OneTo(1), Inclusion(0:0.5:0.5))
            @test axes(QuasiAdjoint(intmat)) == reverse(axes(intmat))
            @test axes(QuasiTranspose(intvec)) == (Base.OneTo(1), Inclusion(0:0.5:0.5))
            @test axes(QuasiTranspose(intmat)) == reverse(axes(intmat))
        end
        @testset "IndexStyle methods" begin
            @test IndexStyle(QuasiAdjoint(intvec)) == IndexCartesian()
            @test IndexStyle(QuasiAdjoint(intmat)) == IndexCartesian()
            @test IndexStyle(QuasiTranspose(intvec)) == IndexCartesian()
            @test IndexStyle(QuasiTranspose(intmat)) == IndexCartesian()
        end
        # vectors and matrices with complex scalar eltype, and their adjoints/transposes
        complexintvec, complexintmat = QuasiVector([1im, 2im],0:0.5:0.5), QuasiMatrix([1im 2im 3im; 4im 5im 6im],(0:0.5:0.5, Base.OneTo(3)))
        tcomplexintvec, tcomplexintmat = QuasiMatrix([1im 2im],(Base.OneTo(1),0:0.5:0.5)), QuasiMatrix([1im 4im; 2im 5im; 3im 6im],(Base.OneTo(3),0:0.5:0.5))
        acomplexintvec, acomplexintmat = conj.(tcomplexintvec), conj.(tcomplexintmat)
        # vectors and matrices with real-vector and real-matrix eltype, and their adjoints/transposes
        intvecvec = QuasiVector([[1, 2], [3, 4]],0:0.5:0.5)
        tintvecvec = QuasiMatrix([[[1 2]] [[3 4]]], (Base.OneTo(1),0:0.5:0.5))
        intmatmat = QuasiMatrix([[[1 2]] [[3  4]] [[ 5  6]];
                    [[7 8]] [[9 10]] [[11 12]]], (0:0.5:0.5,Base.OneTo(3)))
        tintmatmat = QuasiMatrix([[hcat([1, 2])] [hcat([7, 8])];
                    [hcat([3, 4])] [hcat([9, 10])];
                    [hcat([5, 6])] [hcat([11, 12])]], (Base.OneTo(3),0:0.5:0.5))
        # vectors and matrices with complex-vector and complex-matrix eltype, and their adjoints/transposes
        complexintvecvec, complexintmatmat = im .* (intvecvec, intmatmat)
        tcomplexintvecvec, tcomplexintmatmat = im .* (tintvecvec, tintmatmat)
        acomplexintvecvec, acomplexintmatmat = conj.(tcomplexintvecvec), conj.(tcomplexintmatmat)
        @testset "getindex methods, elementary" begin
            # implicitly test elementary definitions, for arrays with concrete real scalar eltype
            @test axes(QuasiAdjoint(intvec)) == axes(tintvec)
            @test QuasiAdjoint(intvec) == tintvec
            @test QuasiAdjoint(intmat) == tintmat
            @test QuasiTranspose(intvec) == tintvec
            @test QuasiTranspose(intmat) == tintmat
            # implicitly test elementary definitions, for arrays with concrete complex scalar eltype
            @test QuasiAdjoint(complexintvec) == acomplexintvec
            @test QuasiAdjoint(complexintmat) == acomplexintmat
            @test QuasiTranspose(complexintvec) == tcomplexintvec
            @test QuasiTranspose(complexintmat) == tcomplexintmat
            # implicitly test elementary definitions, for arrays with concrete real-array eltype
            @test QuasiAdjoint(intvecvec) == tintvecvec
            @test QuasiAdjoint(intmatmat) == tintmatmat
            @test QuasiTranspose(intvecvec) == tintvecvec
            @test QuasiTranspose(intmatmat) == tintmatmat
            # implicitly test elementary definitions, for arrays with concrete complex-array type
            @test QuasiAdjoint(complexintvecvec) == acomplexintvecvec
            @test QuasiAdjoint(complexintmatmat) == acomplexintmatmat
            @test QuasiTranspose(complexintvecvec) == tcomplexintvecvec
            @test QuasiTranspose(complexintmatmat) == tcomplexintmatmat
        end
        @testset "getindex(::AdjOrTransVec, ::Colon, ::AbstractArray{Int}) methods that preserve wrapper type" begin
            # for arrays with concrete scalar eltype
            @test QuasiAdjoint(intvec)[:, [0.0, 0.5]] == Adjoint(parent(intvec))
            @test QuasiTranspose(intvec)[:, [0.0, 0.5]] == Transpose(parent(intvec))
            @test QuasiAdjoint(complexintvec)[:, [0.0, 0.5]] == Adjoint(parent(complexintvec))
            @test QuasiTranspose(complexintvec)[:, [0.0, 0.5]] == Transpose(parent(complexintvec))
            # for arrays with concrete array eltype
            @test QuasiAdjoint(intvecvec)[:, [0.0, 0.5]] == Adjoint(parent(intvecvec))
            @test QuasiTranspose(intvecvec)[:, [0.0, 0.5]] == Transpose(parent(intvecvec))
            @test QuasiAdjoint(complexintvecvec)[:, [0.0, 0.5]] == Adjoint(parent(complexintvecvec))
            @test QuasiTranspose(complexintvecvec)[:, [0.0, 0.5]] == Transpose(parent(complexintvecvec))
        end
        @testset "getindex(::AdjOrTransVec, ::Colon, ::Colon) methods that preserve wrapper type" begin
            # for arrays with concrete scalar eltype
            @test QuasiAdjoint(intvec)[:, :] == QuasiAdjoint(intvec)
            @test QuasiTranspose(intvec)[:, :] == QuasiTranspose(intvec)
            @test QuasiAdjoint(complexintvec)[:, :] == QuasiAdjoint(complexintvec)
            @test QuasiTranspose(complexintvec)[:, :] == QuasiTranspose(complexintvec)
            # for arrays with concrete array elype
            @test QuasiAdjoint(intvecvec)[:, :] == QuasiAdjoint(intvecvec)
            @test QuasiTranspose(intvecvec)[:, :] == QuasiTranspose(intvecvec)
            @test QuasiAdjoint(complexintvecvec)[:, :] == QuasiAdjoint(complexintvecvec)
            @test QuasiTranspose(complexintvecvec)[:, :] == QuasiTranspose(complexintvecvec)
        end
        @testset "getindex(::AdjOrTransVec, ::Colon, ::Int) should preserve wrapper type on result entries" begin
            # for arrays with concrete scalar eltype
            @test QuasiAdjoint(intvec)[:, 0.5] == intvec[0.5:0.5]
            @test QuasiTranspose(intvec)[:, 0.5] == intvec[0.5:0.5]
            @test QuasiAdjoint(complexintvec)[:, 0.5] == conj.(complexintvec[0.5:0.5])
            @test QuasiTranspose(complexintvec)[:, 0.5] == complexintvec[0.5:0.5]
            # for arrays with concrete array eltype
            @test QuasiAdjoint(intvecvec)[:, 0.5] == Adjoint.(intvecvec[0.5:0.5])
            @test QuasiTranspose(intvecvec)[:, 0.5] == Transpose.(intvecvec[0.5:0.5])
            @test QuasiAdjoint(complexintvecvec)[:, 0.5] == Adjoint.(complexintvecvec[0.5:0.5])
            @test QuasiTranspose(complexintvecvec)[:, 0.5] == Transpose.(complexintvecvec[0.5:0.5])
        end
        @testset "setindex! methods" begin
            # for vectors with real scalar eltype
            @test (wv = QuasiAdjoint(copy(intvec));
                    wv === setindex!(wv, 3, 1, 0.5) &&
                    wv == setindex!(copy(tintvec), 3, 1, 0.5)    )
            @test (wv = QuasiTranspose(copy(intvec));
                    wv === setindex!(wv, 4, 1, 0.5) &&
                    wv == setindex!(copy(tintvec), 4, 1, 0.5)    )
            # for matrices with real scalar eltype
            @test (wA = QuasiAdjoint(copy(intmat));
                    wA === setindex!(wA, 7, 3, 0.0) &&
                    wA == setindex!(copy(tintmat), 7, 3, 0.0)    )
            @test (wA = QuasiTranspose(copy(intmat));
                    wA === setindex!(wA, 7, 3, 0.0) &&
                    wA == setindex!(copy(tintmat), 7, 3, 0.0)    )
            # for vectors with complex scalar eltype
            @test (wz = QuasiAdjoint(copy(complexintvec));
                    wz === setindex!(wz, 3im, 1, 0.5) &&
                    wz == setindex!(copy(acomplexintvec), 3im, 1, 0.5)   )
            @test (wz = QuasiTranspose(copy(complexintvec));
                    wz === setindex!(wz, 4im, 1, 0.5) &&
                    wz == setindex!(copy(tcomplexintvec), 4im, 1, 0.5)   )
            # for  matrices with complex scalar eltype
            @test (wZ = QuasiAdjoint(copy(complexintmat));
                    wZ === setindex!(wZ, 7im, 3, 0.0) &&
                    wZ == setindex!(copy(acomplexintmat), 7im, 3, 0.0)   )
            @test (wZ = QuasiTranspose(copy(complexintmat));
                    wZ === setindex!(wZ, 7im, 3, 0.0) &&
                    wZ == setindex!(copy(tcomplexintmat), 7im, 3, 0.0)   )
            # for vectors with concrete real-vector eltype
            @test (wv = QuasiAdjoint(copy(intvecvec));
                    wv === setindex!(wv, Adjoint([5, 6]), 1, 0.5) &&
                    wv == setindex!(copy(tintvecvec), [5 6], 1, 0.5))
            @test (wv = QuasiTranspose(copy(intvecvec));
                    wv === setindex!(wv, Transpose([5, 6]), 1, 0.5) &&
                    wv == setindex!(copy(tintvecvec), [5 6], 1, 0.5))
            # for matrices with concrete real-matrix eltype
            @test (wA = QuasiAdjoint(copy(intmatmat));
                    wA === setindex!(wA, Adjoint([13 14]), 3, 0.5) &&
                    wA == setindex!(copy(tintmatmat), hcat([13, 14]), 3, 0.5))
            @test (wA = QuasiTranspose(copy(intmatmat));
                    wA === setindex!(wA, Transpose([13 14]), 3, 0.5) &&
                    wA == setindex!(copy(tintmatmat), hcat([13, 14]), 3, 0.5))
            # for vectors with concrete complex-vector eltype
            @test (wz = QuasiAdjoint(copy(complexintvecvec));
                    wz === setindex!(wz, Adjoint([5im, 6im]), 1, 0.5) &&
                    wz == setindex!(copy(acomplexintvecvec), [-5im -6im], 1, 0.5))
            @test (wz = QuasiTranspose(copy(complexintvecvec));
                    wz === setindex!(wz, QuasiTranspose([5im, 6im]), 1, 0.5) &&
                    wz == setindex!(copy(tcomplexintvecvec), [5im 6im], 1, 0.5))
            # for matrices with concrete complex-matrix eltype
            @test (wZ = QuasiAdjoint(copy(complexintmatmat));
                    wZ === setindex!(wZ, Adjoint([13im 14im]), 3, 0.0) &&
                    wZ == setindex!(copy(acomplexintmatmat), hcat([-13im, -14im]), 3, 0.0))
            @test (wZ = QuasiTranspose(copy(complexintmatmat));
                    wZ === setindex!(wZ, Transpose([13im 14im]), 3, 0.0) &&
                    wZ == setindex!(copy(tcomplexintmatmat), hcat([13im, 14im]), 3, 0.0))
        end
    end

    @testset "QuasiAdjoint and Transpose convert methods that convert underlying storage" begin
        intvec, intmat = QuasiVector([1, 2],0:0.5:0.5), QuasiMatrix([1 2 3; 4 5 6],(0:0.5:0.5, Base.OneTo(3))) 
        @test convert(QuasiAdjoint{Float64,QuasiVector{Float64,typeof(intvec.axes)}}, QuasiAdjoint(intvec))::QuasiAdjoint{Float64,QuasiVector{Float64,typeof(intvec.axes)}} == QuasiAdjoint(intvec)
        @test convert(QuasiAdjoint{Float64,QuasiMatrix{Float64,typeof(intmat.axes)}}, QuasiAdjoint(intmat))::QuasiAdjoint{Float64,QuasiMatrix{Float64,typeof(intmat.axes)}} == QuasiAdjoint(intmat)
        @test convert(QuasiTranspose{Float64,QuasiVector{Float64,typeof(intvec.axes)}}, QuasiTranspose(intvec))::QuasiTranspose{Float64,QuasiVector{Float64,typeof(intvec.axes)}} == QuasiTranspose(intvec)
        @test convert(QuasiTranspose{Float64,QuasiMatrix{Float64,typeof(intmat.axes)}}, QuasiTranspose(intmat))::QuasiTranspose{Float64,QuasiMatrix{Float64,typeof(intmat.axes)}} == QuasiTranspose(intmat)
    end

    @testset "Adjoint and Transpose similar methods" begin
    intvec, intmat = QuasiVector([1, 2],0:0.5:0.5), QuasiMatrix([1 2 3; 4 5 6],(0:0.5:0.5, Base.OneTo(3))) 
        # similar with no additional specifications, vector (rewrapping) semantics
        @test size(similar(QuasiAdjoint(intvec))::QuasiAdjoint{Int,typeof(intvec)}) == size(QuasiAdjoint(intvec))
        @test size(similar(QuasiTranspose(intvec))::QuasiTranspose{Int,typeof(intvec)}) == size(QuasiTranspose(intvec))
        # similar with no additional specifications, matrix (no-rewrapping) semantics
        @test size(similar(QuasiAdjoint(intmat))::QuasiMatrix{Int,typeof(reverse(intmat.axes))}) == size(QuasiAdjoint(intmat))
        @test size(similar(QuasiTranspose(intmat))::QuasiMatrix{Int,typeof(reverse(intmat.axes))}) == size(QuasiTranspose(intmat))
        # similar with element type specification, vector (rewrapping) semantics
        @test size(similar(QuasiAdjoint(intvec), Float64)::QuasiAdjoint{Float64,QuasiVector{Float64,typeof(intvec.axes)}}) == size(QuasiAdjoint(intvec))
        @test size(similar(QuasiTranspose(intvec), Float64)::QuasiTranspose{Float64,QuasiVector{Float64,typeof(intvec.axes)}}) == size(QuasiTranspose(intvec))
        # similar with element type specification, Quasimatrix (no-rewrapping) semantics
        @test size(similar(QuasiAdjoint(intmat), Float64)::QuasiMatrix{Float64,typeof(reverse(intmat.axes))}) == size(QuasiAdjoint(intmat))
        @test size(similar(QuasiTranspose(intmat), Float64)::QuasiMatrix{Float64,typeof(reverse(intmat.axes))}) == size(QuasiTranspose(intmat))
    end

    @testset "Adjoint and Transpose parent methods" begin
        intvec, intmat = QuasiVector([1, 2],0:0.5:0.5), QuasiMatrix([1 2 3; 4 5 6],(0:0.5:0.5, Base.OneTo(3))) 
        @test parent(QuasiAdjoint(intvec)) === intvec
        @test parent(QuasiAdjoint(intmat)) === intmat
        @test parent(QuasiTranspose(intvec)) === intvec
        @test parent(QuasiTranspose(intmat)) === intmat
    end
end