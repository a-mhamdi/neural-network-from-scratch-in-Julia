module ActivationFunctions

export sigmoid, softmax, tanh, linear, relu, leaky_relu # activation functions
export sigmoid_prime, softmax_prime, tanh_prime, linear_prime, relu_prime, leaky_relu_prime # their corresponding derivatives

## SIGMOID: σ(x)
sigmoid(x) = 1 / (1 + exp(-x))
sigmoid_prime(x) = sigmoid(x) * (1 - sigmoid(x))

## SOFTMAX
softmax(x) = exp.(x) ./ sum(exp.(x))
softmax_prime(x) = softmax(x) .* (1 .- softmax(x))

## TANH
tanh(x) = tanh(x)
tanh_prime(x) = 1 - tanh(x)^2

## Linear
linear(x) = x
linear_prime(x) = 1

## RELU
relu(x) = max(0, x)
relu_prime(x) = ifelse(x > 0, 1, 0)

## LEAKY RELU
leaky_relu(x) = ifelse(x > 0, x, 0.01 * x)
leaky_relu_prime(x) = ifelse(x > 0, 1, 0.01)

end # END