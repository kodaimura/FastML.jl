function sample_linear_data(f = x -> x)
    X = collect(-3:0.1f0:3) .+ rand(Float32, length(-3:0.1f0:3))
    y = f.(X) .+ rand(Float32, length(X))

    X = reshape(X, 61, 1)'
    y = reshape(y, 61, 1)'
    return X, y
end