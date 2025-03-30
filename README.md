# FastML

[![Build Status](https://github.com/kodaimura/FastML.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kodaimura/FastML.jl/actions/workflows/CI.yml?query=branch%3Amain)

FastML は、Flux.jl のモデルをより手軽に学習できるようにしたライブラリです。
線形回帰、ロジスティック回帰、ニューラルネットワーク等の一般的な機械学習を簡潔に実装できます。

| 対応手法　　　　　　　   | 説明                                      | FastML Trainer                 |
|----------------------|------------------------------------------|--------------------------------|
| 線形回帰（単回帰）     | 直線でデータを近似する回帰手法        | `RegressorTrainer(:linear)` |
| 線形回帰（重回帰）     | 平面でデータを近似する回帰手法        | `RegressorTrainer(:multiple_linear)` |
| 多項式回帰            | 特徴量を多項式にして非線形の関係を学習 | `RegressorTrainer(:polynomial)` |
| ニューラルネットワーク  | 非線形関係にも対応できる回帰手法   | `RegressorTrainer(:neural_network)` |
| 二項ロジスティック回帰   | 2クラスの分類手法                 | `BinaryClassifierTrainer(:logistic)` |
| 多項ロジスティック回帰   | 3クラス以上の分類手法              | `SoftmaxClassifierTrainer(:logistic)` |
| ニューラルネットワーク    | 非線形境界にも対応できる分類手法 | `SoftmaxClassifierTrainer(:neural_network)` |

## インストール

```julia
using Pkg
Pkg.add(url="https://github.com/myname/FastML.jl")
```

## 使い方
### 線形回帰（単回帰）

```julia
using FastML, Flux

# データ用意 X::Matrix(特徴量,サンプル数）y::Matrix(1,サンプル数)
X, y = sample_linear_regression_data(x -> 3x + 5)

# 訓練データ・検証データに分ける
X_train, y_train, X_test, y_test = split_train_test(X, y; test_size = 0.2)

# Fluxモデル作成
model = Dense(1 => 1)

# FastML Trainer作成
trainer = RegressorTrainer(:linear)

# 学習
train!(trainer, model, X_train, y_train)

# モデル評価(R2スコア)
@show r2(model, X_train, y_train)
@show r2(model, X_test, y_test)
```

### 線形回帰（重回帰）

```julia
X, y = sample_multiple_linear_regression_data(x -> 3x[1] + 2x[2] - x[3] + 4x[4] - 2x[5] + 1)
X_train, y_train, X_test, y_test = split_train_test(X, y)

model = Dense(5 => 1)
trainer = RegressorTrainer(:multiple_linear)

train!(trainer, model, X_train, y_train)

@show r2(model, X_train, y_train)
@show r2(model, X_test, y_test)
```

### 多項式回帰

```julia
X, y = sample_polynomial_regression_data(x -> 2 + 3x + 5x^2 - 3x^3)
X_train, y_train, X_test, y_test = split_train_test(X, y)

model = Dense(3 => 1)
trainer = RegressorTrainer(:polynomial; learning_rate = 0.0003)

train!(trainer, model, X_train, y_train)

@show r2(model, X_train, y_train)
@show r2(model, X_test, y_test)
```

### ニューラルネット回帰

```julia
X, y = ....
X_train, y_train, X_test, y_test = split_train_test(X, y)

model = Chain(Dense(5 => 20, relu), Dense(20 => 1))
trainer = RegressorTrainer(:neural_network; learning_rate = 0.001, max_epochs = 1000)

train!(trainer, model, X_train, y_train)

@show r2(model, X_train, y_train)
@show r2(model, X_test, y_test)
```

### 二項ロジスティック回帰

```julia
X, y = sample_binary_classification_data(3; n_samples = 100)
X_train, y_train, X_test, y_test = split_train_test(X, y)

model = Chain(Dense(3 => 1, sigmoid))
trainer = BinaryClassifierTrainer(:logistic; learning_rate = 0.003, max_epochs = 5000)

train!(trainer, model, X_train, y_train)

# モデル評価(正答率)
@show accuracy(model, X_test, y_test)
```

### 多項ロジスティック回帰

```julia
classes = [1,2,3]
X, y = sample_classification_data(classes, 2)
X_train, y_train, X_test, y_test = split_train_test(X, y)

model = Chain(Dense(2 => 3), softmax)
trainer = SoftmaxClassifierTrainer(:logistic; learning_rate = 0.01, max_epochs = 5000)

train!(trainer, model, X_train, y_train, classes) #classes必須

@show accuracy(model, X_test, y_test, classes) #classes必須
```

### ニューラルネットワーク分類

```julia
classes = [1, 2, 3, 4, 5]
X, y = sample_classification_data(classes, 3)
X_train, y_train, X_test, y_test = split_train_test(X, y)

model = Chain(Dense(3 => 20, relu), Dense(20 => 5), softmax)
trainer = SoftmaxClassifierTrainer(:neural_network; learning_rate = 0.05, max_epochs = 5000)

train!(trainer, model, X_train, y_train, classes) #classes必須

@show accuracy(model, X_test, y_test, classes) #classes必須
```

### 正則化

```julia
trainer = RegressorTrainer(:linear, :l1; lambda1 = 0.001)
trainer = RegressorTrainer(:linear, :l2; lambda2 = 0.001)
trainer = RegressorTrainer(:linear, :elastic_net; lambda1 = 0.001, lambda2 = 0.001)
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.