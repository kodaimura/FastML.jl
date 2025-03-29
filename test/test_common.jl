function sample_linear_data(f::Function = x -> x; samples = 100, x_min = -3.0, x_max = 3.0)
    X = rand(Float64, samples) * (x_max - x_min) .+ x_min
    y = f.(X) .+ rand(Float32, samples)
    
    X = reshape(X, samples, 1)'
    y = reshape(y, samples, 1)'
    
    return X, y
end

function sample_multiple_linear_data(f::Function = x -> x[1] + x[2]; samples = 100, x_min = -3.0, x_max = 3.0)
    features = dimension(f)

    X = rand(Float64, samples, features) * (x_max - x_min) .+ x_min
    y = [f(x) for x in eachrow(X)] .+ rand(Float32, samples)

    X = reshape(X, samples, features)'
    y = reshape(y, samples, 1)'
    
    return X, y
end

#function sample_polynomial_data(f::Function = x -> x^2, x_range::AbstractRange{T} = -3:0.1f0:3) where T
#    n = length(x_range)
#    X = x_range .+ rand(Float32, n)
#    y = f.(X) .+ rand(Float32, n)
#
#    X = reshape(X, n, 1)'
#    y = reshape(y, n, 1)'
#    return X, y
#end

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