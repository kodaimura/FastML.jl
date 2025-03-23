using FastML
using Test

@testset "FastML.jl" begin
    f(x) = 3x + 5
    data = [([x + rand(Float32)], f(x) + rand(Float32)) for x in -3:0.1f0:3]
    x = hcat([d[1] for d in data]...)
    y = hcat([d[2] for d in data]...)

    x_train, y_train, x_test, y_test = split_data(x, y, 0.2)

    model = LinearRegression(1;learning_rate=0.01,max_epochs=10000,tolerance=1e-6)
    train!(model, x_train, y_train)

    println(r2(model, x_train, y_train))
    println(r2(model, x_test, y_test))

    plot(model, x, y; save_path="linear_regression.png")
end
