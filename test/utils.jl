using Test
using Random


@testset "split_train_test" begin
    X = collect(reshape(1:20, 5, 4))
    y = reshape([0, 1, 0, 1], 1, 4)
    X_train, X_test, y_train, y_test = split_train_test(X, y; test_size=0.5, shuffle=false)

    @testset "訓練データのサイズ" begin
        @test size(X_train, 2) == 2
        @test length(y_train) == 2
    end

    @testset "テストデータのサイズ" begin
        @test size(X_test, 2) == 2
        @test length(y_test) == 2
    end

    @testset "シャッフルなしの順序" begin
        @test X_train[:, 1] == X[:, 1]
        @test X_test[:, 1] == X[:, 3]
    end

    @testset "シャッフルありの動作確認" begin
        X_train_s, X_test_s, y_train_s, y_test_s = split_train_test(X, y; test_size=0.5, shuffle=true, seed=42)
        X_train_s2, X_test_s2, y_train_s2, y_test_s2 = split_train_test(X, y; test_size=0.5, shuffle=true, seed=42)

        @testset "シードを固定した場合の再現性" begin
            @test X_train_s == X_train_s2
            @test X_test_s == X_test_s2
            @test y_train_s == y_train_s2
            @test y_test_s == y_test_s2
        end
    end
end