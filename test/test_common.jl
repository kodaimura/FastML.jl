function sample_linear_data(f::Function = x -> x; samples = 100, x_min = -3.0, x_max = 3.0)
    X = rand(Float64, samples, 1) * (x_max - x_min) .+ x_min
    y = f.(X) .+ rand(Float32, samples)
    
    X = reshape(X, samples, 1)'
    y = reshape(y, samples, 1)'
    
    return X, y
end

function sample_multiple_linear_data(f::Function = x -> x[1] + x[2]; samples = 100, x_min = -3.0, x_max = 3.0)
    dim = dimension(f)

    X = rand(Float64, samples, dim) * (x_max - x_min) .+ x_min
    y = [f(x) for x in eachrow(X)] .+ rand(Float32, samples)

    X = reshape(X, samples, dim)'
    y = reshape(y, samples, 1)'
    
    return X, y
end

function sample_polynomial_data(f::Function = x -> x + x^2; samples = 100, x_min = -3.0, x_max = 3.0)
    deg = degree(f)

    X = rand(Float64, samples, 1) * (x_max - x_min) .+ x_min
    y = f.(X) .+ rand(Float32, samples) * deg
    X_poly = [x ^ i for x in vec(X), i in 1:deg]

    X_poly = reshape(X_poly, samples, deg)'
    y = reshape(y, samples, 1)'

    return X_poly, y
end

function degree(f::Function)
    x_vals = collect(0:60)
    y_vals = f.(x_vals)

    for degree in 1:61
        diff_vals = y_vals
        for _ in 1:degree
            diff_vals = diff(diff_vals)
        end
        
        if all(x -> x == 0, diff_vals)
            return degree - 1
        end
    end
    error("Degree > 60 not supported")
end

function dimension(f::Function)
    for dimension in 1:60
        try
            f(rand(Float64, dimension))
            return dimension 
        catch e
            continue
        end
    end
    error("Dimension > 60 not supported")
end


function sample_classification_data(classes::Int = 2, features::Int = 2; samples = 100, x_min = -3.0, x_max = 3.0)
    centers = [(rand(Float32, features) .* (x_max - x_min) .+ x_min) for _ in 1:classes]
    
    X = []
    y = []
    for class in 1:classes
        mean = centers[class]  # クラスごとの中心
        cov = I(features) * 0.2  # n×n の共分散行列（各次元の分散を0.5に）
        dist = MvNormal(mean, cov)  # n 次元正規分布
        
        num_samples = samples ÷ classes
        if class == classes
            num_samples += samples % classes
        end

        X_class = rand(dist, num_samples)
        y_class = fill(class, num_samples)
        
        append!(X, eachcol(X_class))
        append!(y, y_class)
    end
    
    X = hcat(X...)
    y = reshape(y, 1, samples)

    return X, y
end