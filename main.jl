using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f)

push!(LOAD_PATH, pwd() * "/src")

using Preprocessing, ActivationFunctions, MLP
using Plots

# Hyperparameters
mutable struct Settings
    epochs::Int
    η::Float64
    batch_size::Int

    Settings(epochs, η, batch_size) = batch_size ≥ 1 ? new(epochs, η, batch_size) : error("Batch size must be greater than 1")
    Settings(epochs, η) = new(epochs, η, 1)
end

hp = Settings(16, .001, 6) 

using Random
# Artificial data
n, α, β = 1024, 1., -.5
x = 2 .* randn(n, 5) .+ 5
y = α .* sum(x, dims=2) .+ β |> ( ano -> reshape(ano, n) )

(x_train, y_train), (x_test, y_test), (x_val, y_val) = data_split(x, y, train_size=.7, val_size=.1)
data_x = data_loader(x_train, hp.batch_size)
data_y = data_loader(y_train, hp.batch_size)

# Model architecture
num_features = size(x)[2]
model = [ # MLP
    Layer(num_features, 4,relu; distribution='N'), 
    Layer(4, 1, relu; distribution='N')
    ]

# Regularizer
regularizer = Regularizer(:L2, .8, 0., 0.) # method, λ, r, dropout

ltrn, ltst = [], []
for epoch in 1:hp.epochs
    for (data_in, data_out) in zip(data_x, data_y)
        # Forward pass
        A, H = FeedForward(model, data_in)
        # Backward pass
        loss, ∇W, ∇b = BackProp(model, A, H, data_in, data_out, regularizer)
        # Update parameters
        for i in 1:lastindex(model)
            model[i].W -= hp.η * ∇W[i]
            model[i].b -= hp.η * ∇b[i]
        end
    end

    ### TRAIN LOSS
    _, trn = FeedForward(model, x_train)
    ŷ = [row[end][end] for row in trn]
    loss = sum( ( y_train .- ŷ ) .^ 2 ) / length(y_train)
    push!(ltrn, loss)

    ### TEST LOSS
    _, tst = FeedForward(model, x_test)
    ŷ = [row[end][end] for row in tst]
    loss = sum( ( y_test .- ŷ ) .^ 2 ) / length(y_test)
    push!(ltst, loss)

    @info "epoch $epoch >>> train loss: $(ltrn[end]) *** test loss: $(ltst[end])"
end

plot(ltrn, xlabel="epoch", ylabel="loss", title="loss values", label="train")
plot!(ltst, label="test")