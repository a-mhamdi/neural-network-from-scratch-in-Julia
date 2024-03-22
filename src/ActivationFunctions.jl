module ActivationFunctions

export sigmoid, sigmoid_prime, softmax, softmax_prime, tanh, tanh_prime, relu, relu_prime

# SIGMOID: σ(x)
sigmoid(x) = 1 / (1 + exp(-x))
sigmoid_prime(x) = sigmoid(x) * (1 - sigmoid(x))

# SOFTMAX
softmax(x) = exp.(x) ./ sum(exp.(x))
softmax_prime(x) = softmax(x) .* (1 .- softmax(x))

# TANH
tanh(x) = tanh(x)
tanh_prime(x) = 1 - tanh(x)^2

# RELU
relu(x) = max(0, x)
relu_prime(x) = ifelse(x > 0, 1, 0)

end # END