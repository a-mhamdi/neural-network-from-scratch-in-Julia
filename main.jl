using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f)

using Revise

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

hp = Settings(32, .01, 1) 

using Random
# Artificial data
n, α, β = 512, 1., -.5
x = 2 .* randn(n) .+ 5
y = α .* x .+ β

(x_train, y_train), (x_test, y_test), (x_val, y_val) = data_split(x, y, train_size=.7, val_size=.1)
data_x = data_loader(x_train, hp.batch_size)
data_y = data_loader(y_test, hp.batch_size)

# Model architecture
model = [Layer(1, 1, relu)] # SLP

# Regularizer
regularizer = Regularizer()

coef = 1.
if regularizer.method == :L1
    # FIXME - L1 regularization 
elseif regularizer.method == :L2
    coef -= hp.η * optimizer.λ / hp.batch_size
else regularizer.method == :ElasticNet
    # FIXME - ElasticNet regularization
end

lvec = []
for epoch in 1:hp.epochs
    for data in zip(data_x, data_y)
        # Forward pass
        a, h = FeedForward(model, data[1])
        # Backward pass
        loss, ∇W, ∇b = BackProp(model, a, h, data[2], regularizer)
        push!(lvec, loss)
        # Update parameters
        for i in 1:lastindex(model)
            model[i].W  = coef * model[i].W - hp.η * ∇W[i]
            model[i].b -= hp.η * ∇b[i]
        end
    end
    loss = sum( ( y_test .- (model[1].W .* x_test .+ model[1].b') ) .^ 2 ) / length(y_test)
    @info "Epoch: $epoch >>> Loss: $loss *** ̂α = $(model[1].W[1]) *** ̂β = $(model[1].b[1])"
end

plot(lvec, legend=false, title="Loss Function J", xlabel="Iterations", ylabel="Loss")