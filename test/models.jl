using Test
using FastML


@testset "Trainer" begin
    trainer = Trainer(:linear_regression, :l1; lambda1=0.1, lambda2=0.2, learning_rate=0.05, max_epochs=500, tolerance=1e-5)

    @testset "フィールド" begin
        @test trainer.model_type == LinearRegression
        @test trainer.reg_type == L1
        @test trainer.lambda1 == 0.1
        @test trainer.lambda2 == 0.2
        @test trainer.learning_rate == 0.05
        @test trainer.max_epochs == 500
        @test trainer.tolerance == 1e-5
    end

    @testset "モデルタイプ" begin
        trainer = Trainer(:linear_regression)
        @test trainer.model_type == LinearRegression

        trainer = Trainer(:multiple_linear_regression)
        @test trainer.model_type == MultipleLinearRegression

        trainer = Trainer(:polynomial_regression)
        @test trainer.model_type == PolynomialRegression

        trainer = Trainer(:logistic_regression)
        @test trainer.model_type == LogisticRegression

        trainer = Trainer(:neural_network)
        @test trainer.model_type == NeuralNetwork

        @test_throws AssertionError Trainer(:invalid_model, :l1)
    end

    @testset "正則化タイプ" begin
        trainer = Trainer(:linear_regression)
        @test trainer.reg_type == None

        trainer = Trainer(:linear_regression, :none)
        @test trainer.reg_type == None

        trainer = Trainer(:linear_regression, :l1)
        @test trainer.reg_type == L1

        trainer = Trainer(:linear_regression, :l2)
        @test trainer.reg_type == L2

        trainer = Trainer(:linear_regression, :elastic_net)
        @test trainer.reg_type == ElasticNet

        @test_throws AssertionError Trainer(:linear_regression, :invalid_reg)
    end
    
end