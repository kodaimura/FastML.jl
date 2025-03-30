using Test
using FastML
using Flux
using Random


@testset "split_train_test" begin
    X = collect(reshape(1:20, 5, 4))
    y = reshape([0, 1, 0, 1], 1, 4)
    X_train, y_train, X_test, y_test = split_train_test(X, y; test_size=0.5, shuffle=false)

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
        X_train_s, y_train_s, X_test_s, y_test_s = split_train_test(X, y; test_size=0.5, shuffle=true, seed=42)
        X_train_s2, y_train_s2, X_test_s2, y_test_s2 = split_train_test(X, y; test_size=0.5, shuffle=true, seed=42)

        @testset "シードを固定した場合の再現性" begin
            @test X_train_s == X_train_s2
            @test X_test_s == X_test_s2
            @test y_train_s == y_train_s2
            @test y_test_s == y_test_s2
        end
    end
end

@testset "sample_linear_regression_data" begin
    @testset "デフォルト" begin
        X, y = sample_linear_regression_data()
        @test size(X, 1) == 1
        @test size(X, 2) == 100
        @test size(y, 1) == 1
        @test size(y, 2) == 100

        @test all(x -> -4 ≤ x ≤ 4, vec(X))
        @test all(y -> -4 ≤ y ≤ 4, vec(y))
    end

    @testset "x -> 3x + 5" begin
        X, y = sample_linear_regression_data(x -> 3x + 5)
        @test size(X, 1) == 1
        @test size(X, 2) == 100
        @test size(y, 1) == 1
        @test size(y, 2) == 100

        @test all(x -> -4 ≤ x ≤ 4, vec(X))
        @test all(y -> -7 ≤ y ≤ 17, vec(y))
    end

    @testset "x -> 3x + 5; n_samples = 200, x_min=-2, x_max=2" begin
        X, y = sample_linear_regression_data(x -> 3x + 5; n_samples = 200, x_min=-2, x_max=2)
        @test size(X, 1) == 1
        @test size(X, 2) == 200
        @test size(y, 1) == 1
        @test size(y, 2) == 200

        @test all(x -> -2 - 1 ≤ x ≤ 2 + 1, vec(X))
        @test all(y -> -4 ≤ y ≤ 14, vec(y))
    end

    @testset "x -> 3x + 5; n_samples = 1" begin
        X, y = sample_linear_regression_data(x -> 3x + 5; n_samples = 1)
        @test size(X, 1) == 1
        @test size(X, 2) == 1
        @test size(y, 1) == 1
        @test size(y, 2) == 1
    end 
end

@testset "sample_multiple_linear_regression_data" begin
    @testset "デフォルト" begin
        X, y = sample_multiple_linear_regression_data()
        @test size(X, 1) == 2
        @test size(X, 2) == 100
        @test size(y, 1) == 1
        @test size(y, 2) == 100

        @test all(x -> -4 ≤ x ≤ 4, vec(X))
        @test all(y -> -8 ≤ y ≤ 8, vec(y))
    end

    @testset "x -> 3x[1] + 2x[2] - x[3] + 1" begin
        X, y = sample_multiple_linear_regression_data(x -> 3x[1] + 2x[2] - x[3] + 1)
        @test size(X, 1) == 3
        @test size(X, 2) == 100
        @test size(y, 1) == 1
        @test size(y, 2) == 100

        @test all(x -> -4 ≤ x ≤ 4, vec(X))
        @test all(y -> -23 ≤ y ≤ 25, vec(y))
    end

    @testset "x -> 3x[1] + 2x[2] - x[3] + 1; n_samples = 200, x_min=-2, x_max=2" begin
        X, y = sample_multiple_linear_regression_data(x -> 3x[1] + 2x[2] - x[3] + 1; n_samples = 200, x_min=-2, x_max=2)
        @test size(X, 1) == 3
        @test size(X, 2) == 200
        @test size(y, 1) == 1
        @test size(y, 2) == 200

        @test all(x -> -3 ≤ x ≤ 3, vec(X))
        @test all(y -> -17 ≤ y ≤ 19, vec(y))
    end

    @testset "x -> 3x[1]; n_samples = 1" begin
        X, y = sample_linear_regression_data(x -> 3x[1]; n_samples = 1)
        @test size(X, 1) == 1
        @test size(X, 2) == 1
        @test size(y, 1) == 1
        @test size(y, 2) == 1
    end 
end

@testset "sample_polynomial_regression_data" begin
    @testset "デフォルト" begin
        X, y = sample_polynomial_regression_data()
        @test size(X, 1) == 2
        @test size(X, 2) == 100
        @test size(y, 1) == 1
        @test size(y, 2) == 100
        
        @test all(x -> -4 ≤ x ≤ 4, X[1, :])
        @test all(x -> 0 ≤ x ≤ 16, X[2, :])
        @test all(y -> -4 ≤ y ≤ 20, vec(y))
    end

    @testset "x -> 2 + 3x + 5x^2 - 3x^3" begin
        X, y = sample_polynomial_regression_data(x -> 2 + 3x + 5x^2 - 3x^3)
        @test size(X, 1) == 3
        @test size(X, 2) == 100
        @test size(y, 1) == 1
        @test size(y, 2) == 100

        @test all(x -> -4 ≤ x ≤ 4, X[1, :])
        @test all(x -> 0 ≤ x ≤ 16, X[2, :])
        @test all(x -> -64 ≤ x ≤ 64, X[3, :])
    end

    @testset "x -> 2 + 3x + 5x^2 - 3x^3; n_samples = 200, x_min=-2, x_max=2" begin
        X, y = sample_polynomial_regression_data(x -> 2 + 3x + 5x^2 - 3x^3; n_samples = 200, x_min=-2, x_max=2)
        @test size(X, 1) == 3
        @test size(X, 2) == 200
        @test size(y, 1) == 1
        @test size(y, 2) == 200
    end

    @testset "x -> 3x[1]; n_samples = 1" begin
        X, y = sample_polynomial_regression_data(x -> 2 + 3x; n_samples = 1)
        @test size(X, 1) == 1
        @test size(X, 2) == 1
        @test size(y, 1) == 1
        @test size(y, 2) == 1
    end 
end

@testset "sample_classification_data" begin
    @testset "デフォルト" begin
        X, y = sample_classification_data()
        @test size(X, 1) == 1
        @test size(X, 2) == 100
        @test size(y, 1) == 1
        @test size(y, 2) == 100

        #@test all(x -> -4 ≤ x ≤ 4, vec(X))
        @test all(y -> y in (0, 1), vec(y))
    end

    @testset "classes = [1,2,3], features = 2" begin
        X, y = sample_classification_data([1,2,3], 2)
        @test size(X, 1) == 2
        @test size(X, 2) == 100
        @test size(y, 1) == 1
        @test size(y, 2) == 100

        #@test all(x -> -4 ≤ x ≤ 4, vec(X))
        @test all(y -> y in (1, 2, 3), vec(y))
    end

    @testset "classes = [2,4,6,8,10], features = 3; n_samples = 200, x_min=-2, x_max=2" begin
        X, y = sample_classification_data([2,4,6,8,10], 3; n_samples = 200, x_min=-2, x_max=2)
        @test size(X, 1) == 3
        @test size(X, 2) == 200
        @test size(y, 1) == 1
        @test size(y, 2) == 200

        #@test all(x -> -3 ≤ x ≤ 3, vec(X))
        @test all(y -> y in (2, 4, 6, 8, 10), vec(y))
    end

    @testset "classes = [0], features = 1; n_samples = 1" begin
        X, y = sample_classification_data([0], 1; n_samples = 1)
        @test size(X, 1) == 1
        @test size(X, 2) == 1
        @test size(y, 1) == 1
        @test size(y, 2) == 1
    end 
end