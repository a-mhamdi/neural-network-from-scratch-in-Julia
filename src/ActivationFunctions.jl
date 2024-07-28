module ActivationFunctions

export sigmoid, softmax, tanh, linear, relu, leaky_relu 

## SIGMOID: σ(x)
sigmoid(x; diff=false) = diff ? sigmoid.(x) .* (1 .- sigmoid.(x)) : 1 ./ (1 .+ exp.(-x))

## SOFTMAX
function softmax(x; diff=false)
    if isempty(size(x))
        x = reshape([x], 1, 1)
    end

    if diff
        z = softmax(x)
        n = length(z)
        J = zeros(n, n)
        for i in 1:n
            for j in 1:n
                J[i, j] = i == j ? z[i] * (1 - z[j]) : -z[i] * z[j]
            end
        end

        return J
    else
        z = zeros(size(x))
        for (ix, col) in enumerate(eachcol(x))
            e = exp.(col .- maximum(col))
            s = sum(e)

            z[:, ix] = e ./ s
        end

        return z
    end
end

## TANH
tanh(x; diff=false) = diff ? 1 - tanh.(x) .^ 2 : tan.h(x)

## Linear
linear(x; diff=false) = diff ? 1 : x

## RELU
relu(x; diff=false) = diff ? ifelse.(x .≥ 0, 1, 0) : max.(0, x)

## LEAKY RELU
leaky_relu(x, alpha; diff=false) = diff ? ifelse.(x .≥ 0, 1, alpha) : ifelse.(x .≥ 0, x, alpha .* x)

end # END