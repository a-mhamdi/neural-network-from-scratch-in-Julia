module Preprocessing

export data_split, data_loader

using Random

# Data split
function data_split(X::Vector{Float64}, y::Vector{Float64}; shuffle=true, train_size::Float64=.8, val_size::Float64=.0)
    n = length(X)
    ntrain = floor(Int, n * train_size)

    test_size = ifelse( val_size == 0., 1. - train_size, 1. - train_size - val_size)
    ntest = floor(Int, n * test_size)

    train, test, val = 1:ntrain, ntrain+1:ntrain+ntest, ntrain+ntest+1:n

    if shuffle
        p = randperm(n)
        X = X[p]
        y = y[p]
    end
    
    X_train, X_test, X_val = X[train], X[test], X[val]
    y_train, y_test, y_val = y[train], y[test], y[val]

    (X_train, y_train), (X_test, y_test), (X_val, y_val)
end

# Data loader
function data_loader(data, batch_size)
    # Calculate the total number of batches
    num_batches = Int(ceil(length(data) / batch_size))
    # Partition data into batches
    batches = [ data[(i-1)*batch_size+1:min(i*batch_size, end)] for i in 1:num_batches ]
    return batches
end

end # END