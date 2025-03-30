using Test
using FastML

@testset "Trainer" begin

    @testset "デフォルト" begin
        trainer = RegressorTrainer(:linear)
        @test trainer.model_type == Linear
        @test trainer.reg_type == None
        @test trainer.lambda1 == 0.0
        @test trainer.lambda2 == 0.0
        @test trainer.learning_rate == 0.01
        @test trainer.max_epochs == 1000
        @test trainer.tolerance == 1e-6
    end

    @testset "カスタム" begin
        trainer = RegressorTrainer(:linear, :l1; lambda1=0.1, lambda2=0.2, learning_rate=0.05, max_epochs=500, tolerance=1e-5)
        @test trainer.model_type == Linear
        @test trainer.reg_type == L1
        @test trainer.lambda1 == 0.1
        @test trainer.lambda2 == 0.2
        @test trainer.learning_rate == 0.05
        @test trainer.max_epochs == 500
        @test trainer.tolerance == 1e-5
    end

    @testset "モデルタイプ" begin
        trainer = RegressorTrainer(:linear)
        @test trainer.model_type == Linear

        trainer = RegressorTrainer(:multiple_linear)
        @test trainer.model_type == MultipleLinear

        trainer = RegressorTrainer(:polynomial)
        @test trainer.model_type == Polynomial

        trainer = RegressorTrainer(:neural_network)
        @test trainer.model_type == NeuralNetwork

        @test_throws AssertionError RegressorTrainer(:invalid_model, :l1)
    end

    @testset "正則化タイプ" begin
        trainer = RegressorTrainer(:linear, :none)
        @test trainer.reg_type == None

        trainer = RegressorTrainer(:linear, :l1)
        @test trainer.reg_type == L1

        trainer = RegressorTrainer(:linear, :l2)
        @test trainer.reg_type == L2

        trainer = RegressorTrainer(:linear, :elastic_net)
        @test trainer.reg_type == ElasticNet

        @test_throws AssertionError RegressorTrainer(:linear, :invalid_reg)
    end
end

@testset "BinaryClassifierTrainer" begin

    @testset "デフォルト" begin
        trainer = BinaryClassifierTrainer(:logistic)
        @test trainer.model_type == BinaryLogistic
        @test trainer.reg_type == None
        @test trainer.lambda1 == 0.0
        @test trainer.lambda2 == 0.0
        @test trainer.learning_rate == 0.01
        @test trainer.max_epochs == 1000
        @test trainer.tolerance == 1e-6
    end

    @testset "カスタム" begin
        trainer = BinaryClassifierTrainer(:logistic, :l1; lambda1=0.1, lambda2=0.2, learning_rate=0.05, max_epochs=500, tolerance=1e-5)
        @test trainer.model_type == BinaryLogistic
        @test trainer.reg_type == L1
        @test trainer.lambda1 == 0.1
        @test trainer.lambda2 == 0.2
        @test trainer.learning_rate == 0.05
        @test trainer.max_epochs == 500
        @test trainer.tolerance == 1e-5
    end

    @testset "モデルタイプ" begin
        trainer = BinaryClassifierTrainer(:logistic)
        @test trainer.model_type == BinaryLogistic
        @test_throws AssertionError BinaryClassifierTrainer(:invalid_model)
    end

    @testset "正則化タイプ" begin
        trainer = BinaryClassifierTrainer(:logistic, :none)
        @test trainer.reg_type == None

        trainer = BinaryClassifierTrainer(:logistic, :l1)
        @test trainer.reg_type == L1

        trainer = BinaryClassifierTrainer(:logistic, :l2)
        @test trainer.reg_type == L2

        trainer = BinaryClassifierTrainer(:logistic, :elastic_net)
        @test trainer.reg_type == ElasticNet

        @test_throws AssertionError BinaryClassifierTrainer(:logistic, :invalid_reg)
    end
end

@testset "SoftmaxClassifierTrainer" begin
    @testset "デフォルト" begin
        trainer = SoftmaxClassifierTrainer(:logistic)
        @test trainer.model_type == SoftmaxLogistic
        @test trainer.reg_type == None
        @test trainer.lambda1 == 0.0
        @test trainer.lambda2 == 0.0
        @test trainer.learning_rate == 0.01
        @test trainer.max_epochs == 1000
        @test trainer.tolerance == 1e-6
    end

    @testset "カスタム" begin
        trainer = SoftmaxClassifierTrainer(:logistic, :l1; lambda1=0.1, lambda2=0.2, learning_rate=0.05, max_epochs=500, tolerance=1e-5)
        @test trainer.model_type == SoftmaxLogistic
        @test trainer.reg_type == L1
        @test trainer.lambda1 == 0.1
        @test trainer.lambda2 == 0.2
        @test trainer.learning_rate == 0.05
        @test trainer.max_epochs == 500
        @test trainer.tolerance == 1e-5
    end

    @testset "モデルタイプ" begin
        trainer = SoftmaxClassifierTrainer(:logistic)
        @test trainer.model_type == SoftmaxLogistic

        trainer = SoftmaxClassifierTrainer(:neural_network)
        @test trainer.model_type == SoftmaxNeuralNetwork

        @test_throws AssertionError SoftmaxClassifierTrainer(:invalid_model)
    end

    @testset "正則化タイプ" begin
        trainer = SoftmaxClassifierTrainer(:logistic, :none)
        @test trainer.reg_type == None

        trainer = SoftmaxClassifierTrainer(:logistic, :l1)
        @test trainer.reg_type == L1

        trainer = SoftmaxClassifierTrainer(:logistic, :l2)
        @test trainer.reg_type == L2

        trainer = SoftmaxClassifierTrainer(:logistic, :elastic_net)
        @test trainer.reg_type == ElasticNet

        @test_throws AssertionError SoftmaxClassifierTrainer(:logistic, :invalid_reg)
    end
end
