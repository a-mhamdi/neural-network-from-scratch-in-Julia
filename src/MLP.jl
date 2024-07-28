module MLP

using ActivationFunctions
using Metrics

export FeedForward, BackProp

export Regularizer, Solver
export Layer, TrainNN, Predict, loss_fct

### N-NET ARCHITECTURE
mutable struct Layer
    W::Array{Float64, 2}
    b::Vector{Float64}
    act::Function
end

function Layer(input_size::Int, output_size::Int, act::Function; distribution::Char='u')
    if act in [relu, leaky_relu] # He initialization
        if distribution == 'u' # uniform
            W = sqrt(6/input_size) .* (2 .* rand(output_size, input_size) .- 1)
        elseif distribution == 'n' # normal
            W = sqrt(2/input_size) .* randn(output_size, input_size)
        else
            @error "Invalid distribution"
        end
    else # Xavier (Glorot) initialization
        if distribution == 'u'
            W = sqrt(6/(input_size + output_size)) .* (2 .* rand(output_size, input_size) .- 1) 
        elseif distribution == 'n'
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
    Regularizer() = new(:none, 0., 0., 0.)
end

### SOLVER: loss, optimizer, η, regularizer
struct Solver
    loss::Union{Symbol, String} # :MSE, :CrossEntropy
    optimizer::Union{Symbol, String} # :SGD, :Adam, :RMSprop
    η::Float64
    regularizer::Regularizer

    Solver(a, b, c, d) = new(a, b, c, d)
    Solver(d::Regularizer) = new(:mse, :sgd, .001, d)
    Solver() = new(:None, :None, 0., Regularizer())
end

### TRAINING
function TrainNN(layers::Vector{Layer}, data_in, data_out, x_val, y_val; solver::Solver)
    opt = solver.optimizer
    ## Forward pass
    data_cache = FeedForward(layers, data_in; solver)
    ## Backward pass
    loss_bp, ∇W, ∇b = BackProp(layers, data_cache, data_out; solver)
    ## Update parameters
    for i in 1:lastindex(layers)
        if (isa(opt, Symbol) && opt == :sgd) || (isa(opt, String) && lowercase(opt) == "sgd")
            layers[i].W -= solver.η .* ∇W[i]
            layers[i].b -= solver.η .* ∇b[i]
        elseif (isa(opt, Symbol) && opt == :adam) || (isa(opt, String) && lowercase(opt) == "adam")
            @warn ("`ADAM` OPTIMIZER: NOT YET IMPLEMENTED. PLEASE USE `SGD` INSTEAD.")
            break
            β1 = 0.9
            β2 = 0.999
            ε = 1e-8
            MW = (β1 .* MW .+ (1 - β1) .* ∇W) ./ (1 - β1^ix)
            Mb = (β1 .* Mb .+ (1 - β1) .* ∇b) ./ (1 - β1^ix)
            VW = (β2 .* VW .+ (1 - β2) .* (∇W .^ 2)) ./ (1 - β2^ix)
            Vb = (β2 .* Vb .+ (1 - β2) .* (∇b .^ 2)) ./ (1 - β2^ix)
            try
                layers[i].W -= solver.η .*(sqrt.(VW) .+ ε) .* MW
                layers[i].b -= solver.η .*(sqrt.(Vb) .+ ε) .* Mb
            catch
                @warn "Complex argument with `adam` solver"
            end
        end
    end
    ## Validation
    ŷ_val = Predict(layers, x_val)
    loss_val = loss_fct(y_val, ŷ_val; loss=solver.loss)

    @info "loss >>> train: $loss_bp *** val: $loss_val"

    ∇W, ∇b
end

### PREDICTION
function Predict(layers::Vector{Layer}, x)
    data_out = FeedForward(layers, x; output=true)

    data_out
end

### FEEDFORWARD
function FeedForward(layers::Vector{Layer}, data_in; solver::Solver=Solver(), output::Bool=false)
    ## CHECK DIMS OF `data_in`
    if isempty(size(data_in))
        data_in = reshape([data_in], 1, 1)
    elseif typeof(size(data_in)) == Tuple{Int64}
        data_in = reshape(data_in, 1, length(data_in))
    end

    dropout = solver.regularizer.dropout
    data_cache = [[], []]
    a, h = [], []
    nrows, _ = size(data_in)
    nouts = length(layers[end].b)
    data_out = zeros(nrows, nouts)
    for i in 1:nrows
        data = data_in[i, :]
        push!(h, data)
        for (ix, l) in enumerate(layers)
            data = l(data)
            len = length(l.b)
            vec = rand(1:len, len)
            indices = 1:Int(floor(dropout * len))
            if ix != length(layers)
                data[vec[indices]] .= 0.
            end
            push!(a, data)
            data = l.act(data)
            if ix != length(layers)
                data[vec[indices]] .= 0.
                data ./= (1 - dropout)
            end
            push!(h, data)
        end

        data_out[i, :] = h[end]
        
        push!(data_cache[1], a); a = []
        push!(data_cache[2], h); h = []
    end

    if output
        return data_out
    else
        return data_cache ## dims = (2, batch_size, first_layer_output_size ... last_layer_output_size)
    end
end

### BACKPROP
function BackProp(layers::Vector{Layer}, data_cache, data_out; solver::Solver)
    
    reg = solver.regularizer

    loss, ∇W, ∇b = 0., [], []

    ∇W = [zeros(size(l.W)) for l in layers]
    ∇b = [zeros(size(l.b)) for l in layers]

    batch_size = size(data_cache[1])[1]

    for i in 1:batch_size
        a, h = data_cache[1][i], data_cache[2][i]
        y = data_out[i, :]

        loss += loss_fct(y, h[end]; loss=solver.loss)

        if (isa(reg.method, Symbol) && reg.method == :l1) || (isa(reg.method, String) && lowercase(reg.method) == "l1") # LASSO 
            for (ix, l) in enumerate(layers)
                loss += reg.λ .* sum(abs.(l.W))
                ∇W[ix] .+= reg.λ .* sign.(l.W)
            end
        elseif (isa(reg.method, Symbol) && reg.method == :l2) || (isa(reg.method, String) && lowercase(reg.method) == "l2") # RIDGE
            for (ix, l) in enumerate(layers)
                loss += reg.λ/2 .* sum(l.W .^ 2)
                ∇W[ix] .+= reg.λ .* l.W
            end
        elseif (isa(reg.method, Symbol) && reg.method == :elasticnet) || (isa(reg.method, String) && lowercase(reg.method) == "elasticnet")
            for (ix, l) in enumerate(layers)
                loss += reg.r * reg.λ .* sum(abs.(l.W)) .+ (1-reg.r) * reg.λ/2 .* sum(l.W .^ 2)
                ∇W[ix] .+= reg.r * reg.λ .* sign.(l.W) .+ (1-reg.r) * reg.λ .* l.W
            end
        end

        len = length(layers)
        ## Compute δ
        if (isa(solver.loss, Symbol) && solver.loss == :binarycrossentropy) || (isa(solver.loss, String) && lowercase(solver.loss) == "binarycrossentropy")
            δ = [layers[end].act(a[end]; diff=true) .* (y ./ (h[end] .+ 1e-10) .- (1 .- y) ./ (1 .- h[end]))]
        elseif layers[end].act == softmax
            δ = [softmax(a[end]; diff=true)' * (y ./ (h[end] .+ 1e-10))]
        else
            δ = [layers[end].act(a[end]; diff=true) .* (y .- h[end])]
        end
        
        for j in len-1:-1:1
            pushfirst!(δ, (layers[j+1].W' * δ[1]) .* layers[j].act(a[j]; diff=true))
        end

        ## Compute gradients
        ∇W += [-δ[k] * h[k]' for k in 1:len]
        ∇b += -δ

        a, h = [], []
    end

    loss /= batch_size
    ∇W  ./= batch_size
    ∇b  ./= batch_size

    loss, ∇W, ∇b
end

end ## END