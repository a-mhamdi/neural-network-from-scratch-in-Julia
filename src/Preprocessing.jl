module Preprocessing

export data_split, data_loader

using Random

# Data split
function data_split(X, y; shuffle=true, split::Float64=0.8)
    n = length(X)
    n_train = floor(Int, n * split)

    if shuffle
        p = randperm(n)
        X = X[p]
        y = y[p]
    end
    
    X_train, X_test = X[1:n_train], X[n_train+1:end]
    y_train, y_test = y[1:n_train], y[n_train+1:end]

    (X_train, y_train), (X_test, y_test)
end

# Data loader
data_loader(X, y, batch_size) = hcat(collect(Iterators.partition(X, batch_size)), collect(Iterators.partition(y, batch_size)))

end # END