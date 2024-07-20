using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f)

push!(LOAD_PATH, pwd() * "/src")

using Revise

using Preprocessing, ActivationFunctions, MLP

## Hyperparameters
mutable struct Settings
    epochs::Int
    batch_size::Int

    Settings(epochs, batch_size) = batch_size ≥ 1 ? new(epochs, batch_size) : error("Batch size must be greater than 1")
    Settings(epochs) = new(epochs, 1)
end

hp = Settings(32, 16) 

using Random
## Artificial data
n, α, β = 512, 1.5, -.5
x = 5 .* randn(n, 3) 
# y = Int.( sum(x, dims=2) .≥ 0) ## classification 
y = α .* sum(x, dims=2) .+ β ## regression

(x_train, y_train), (x_test, y_test), (x_val, y_val) = data_split(x, y, train_size=.7, val_size=.1)
data_x = data_loader(x_train, hp.batch_size)
data_y = data_loader(y_train, hp.batch_size)

## Model architecture
num_features = size(x)[2]
model = [ # MLP
    Layer(num_features, 4, relu; distribution='N'),
    Layer(4, 7, relu; distribution='N'),
    Layer(7, 1, relu; distribution='N')
    ]

## Regularizer
regularizer = Regularizer(:ElasticNet, .2, .8, .6) # method, λ, r, dropout

## Solver
solver = Solver(:MAE, :SGD, 1e-2, regularizer)

ltrn, ltst = [], []

for epoch in 1:hp.epochs
    for (data_in, data_out) in zip(data_x, data_y)
        TrainNN(model, data_in, data_out, x_val, y_val; solver)
    end

    ### TRAIN LOSS
    ŷ_trn = Predict(model, x_train)
    loss = Loss(y_train, ŷ_trn; loss=solver.loss) 
    push!(ltrn, loss)

    ### TEST LOSS
    ŷ_tst = Predict(model, x_test)
    loss = Loss(y_test, ŷ_tst; loss=solver.loss)
    push!(ltst, loss)

    println("=============== SUMMARY @ EPOCH $epoch =================")
    println("train loss: $(ltrn[end]) *** test loss: $(ltst[end])")
end

using Plots
plot(ltrn, label="train", xlabel="epoch", ylabel="loss", title="loss values")
plot!(ltst, label="test")