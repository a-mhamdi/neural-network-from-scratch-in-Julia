module MLP

using LinearAlgebra
using ActivationFunctions

export FeedForward, BackProp

export Regularizer, Solver
export Layer, TrainNN, Predict, Loss

### N-NET ARCHITECTURE
mutable struct Layer
    W::Array{Float64, 2}
    b::Vector{Float64}
    act::Function
end

function Layer(input_size::Int, output_size::Int, act::Function; distribution::Char='U')
    if act in [relu, leaky_relu] # He initialization
        if distribution == 'U' # uniform
            W = sqrt(6/input_size) .* (2 .* rand(output_size, input_size) .- 1)
        elseif distribution == 'N' # normal
            W = sqrt(2/input_size) .* randn(output_size, input_size)
        else
            @error "Invalid distribution"
        end
    else # Xavier (Glorot) initialization
        if distribution == 'U'
            W = sqrt(6/(input_size + output_size)) .* (2 .* rand(output_size, input_size) .- 1) 
        elseif distribution == 'N'
            W = sqrt(2/(input_size + output_size)) .* randn(output_size, input_size)
        else
            @error "Invalid distribution"
        end
    end
    b = zeros(output_size)
    Layer(W, b, act) 
end

function (l::Layer)(x) # ::Array{Float64, 2})
    l.W * x .+ l.b
end

### REGULARIZER: method, λ, r, dropout
struct Regularizer
    method::Union{Symbol, String} # :L1, :L2, :ElasticNet
    λ::Float64 
    r::Float64 
    dropout::Float64 

    Regularizer(a, b, c, d) = new(a, b, c, d)
    Regularizer() = new(:None, 0., 0., 0.)
end

### SOLVER: loss, optimizer, η, regularizer
struct Solver
    loss::Union{Symbol, String} # :MSE, :CrossEntropy
    optimizer::Union{Symbol, String} # :SGD, :Adam, :RMSprop
    η::Float64
    regularizer::Regularizer

    Solver(a, b, c, d) = new(a, b, c, d)
    Solver(d::Regularizer) = new(:MSE, :SGD, .001, d)
    Solver() = new(:None, :None, 0., Regularizer())
end

### LOSS
function Loss(y, ŷ; loss::Union{Symbol,String}=:MSE)
    if (isa(loss, Symbol) && loss == :MAE) || (isa(loss, String) && lowercase(loss) == "mae") # mean absolute error
        ls = sum(abs.(y .- ŷ)) ./ length(y)
    elseif (isa(loss, Symbol) && loss == :MSE) || (isa(loss, String) && lowercase(loss) == "mse") # mean squared error
        ls = sum((y .- ŷ) .^ 2) ./ length(y)
    elseif (isa(loss, Symbol) && loss == :CrossEntropy) || (isa(loss, String) && lowercase(loss) == "crossentropy")
        if typeof(size(y)) == Tuple{Int64}
        ls = sum(-y .* log.(ŷ) .- (1 .- y) .* log.(1 .- ŷ)) ./ length(y) # binary cross entropy
        else
        ls = sum(-y .* log.(ŷ)) ./ length(y) # categorical cross entropy
        end
    end

    ls
end

### FEEDFORWARD
function FeedForward(layers::Vector{Layer}, data_in; solver::Solver=Solver())
    ## CHECK DIMS OF `data_in`
    if isempty(size(data_in))
        data_in = reshape([data_in], 1, 1)
    elseif typeof(size(data_in)) == Tuple{Int64}
        data_in = reshape(data_in, length(data_in), 1)
    end

    dropout = solver.regularizer.dropout
    A, a = [], []
    H, h = [], []
    nrows, _ = size(data_in)
    for i in 1:nrows
        data = data_in[i, :]
        for (ix, l) in enumerate(layers)
            len = length(l.b)
            vec = rand(1:len, len)
            indices = 1:Int(floor(dropout * len))
            data = l(data)
            if ix != length(layers)
                data[vec[indices]] .= 0.
            end
            push!(a, data)
            data = l.act.(data)
            if ix != length(layers)
                data ./= (1 - dropout)
            end
            push!(h, data)
        end
        push!(A, a); a = []
        push!(H, h); h = []
    end

    A, H ## dims = (batch_size, first_layer_output_size ... last_layer_output_size)
end

### BACKPROP
function BackProp(layers::Vector{Layer}, A, H, data_in, data_out; solver::Solver)
    
    reg = solver.regularizer

    loss, ∇W, ∇b = 0., [], []

    ∇W = [zeros(size(l.W)) for l in layers]
    ∇b = [zeros(size(l.b)) for l in layers]

    batch_size = size(A)[1]

    for i in 1:batch_size
        a = A[i]
        h = H[i]
        pushfirst!(h, data_in[i, :])
        y = data_out[i]

        loss += Loss(y, h[end]; loss=solver.loss)

        if (isa(reg.method, Symbol) && reg.method == :L1) || (isa(reg.method, String) && lowercase(reg.method) == "l1") # LASSO 
            for (ix, l) in enumerate(layers)
                loss += reg.λ .* sum(abs.(l.W))
                ∇W[ix] .+= reg.λ .* sign.(l.W)
            end
        elseif (isa(reg.method, Symbol) && reg.method == :L2) || (isa(reg.method, String) && lowercase(reg.method) == "l2") # RIDGE
            for (ix, l) in enumerate(layers)
                loss += reg.λ/2 .* sum(l.W .^ 2)
                ∇W[ix] .+= reg.λ .* l.W
            end
        elseif (isa(reg.method, Symbol) && reg.method == :ElasticNet) || (isa(reg.method, String) && lowercase(reg.method) == "elasticnet")
            for (ix, l) in enumerate(layers)
                loss += reg.r * reg.λ .* sum(abs.(l.W)) .+ (1-reg.r) * reg.λ/2 .* sum(l.W .^ 2)
                ∇W[ix] .+= reg.r * reg.λ .* sign.(l.W) .+ (1-reg.r) * reg.λ .* l.W
            end
        end

        ## Compute δ
        act_prime = eval(Symbol(layers[end].act, "_prime")).(a[end])
        δ = [(y .- h[end]) .* act_prime]
        for j in length(layers)-1:-1:1
            act_prime = eval(Symbol(layers[j].act, "_prime")).(a[j])
            pushfirst!(δ, act_prime .* (layers[j+1].W' * δ[1]))
        end

        ## Compute gradients
        ∇W += [-δ[k] * h[k]' for k in 1:length(layers)]
        ∇b += -δ

        a, h = [], []
    end

    loss /= batch_size
    ∇W  ./= batch_size
    ∇b  ./= batch_size

    loss, ∇W, ∇b
end

### TRAINING
function TrainNN(layers::Vector{Layer}, data_in, data_out, x_val, y_val; solver::Solver)
    ## Forward pass
    A, H = FeedForward(layers, data_in; solver)
    ## Backward pass
    loss_bp, ∇W, ∇b = BackProp(layers, A, H, data_in, data_out; solver)
    ## Update parameters
    for i in 1:lastindex(layers)
        layers[i].W -= solver.η .* ∇W[i]
        layers[i].b -= solver.η .* ∇b[i]
    end
    ## Validation
    ŷ_val = Predict(layers, x_val)
    loss_val = Loss(y_val, ŷ_val; loss=solver.loss)

    @info "loss >>> train: $loss_bp *** val: $loss_val"

    ∇W, ∇b
end

### PREDICTION
function Predict(layers::Vector{Layer}, x)
    _, H = FeedForward(layers, x)
    output = [row[end][end] for row in H]

    output
end

end ## END



