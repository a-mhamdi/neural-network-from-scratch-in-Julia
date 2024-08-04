module Preprocessing

export data_split, data_loader


using Random

## DATA SPLITTER
function data_split(x, y; shuffle=true, train_size::Float64=.8, val_size::Float64=.0)
    nrows = size(x)[1]
    ntrain = floor(Int, nrows * train_size)

    test_size = ifelse( val_size == 0., 1. - train_size, 1. - train_size - val_size)
    ntest = floor(Int, nrows * test_size)

    train, test, val = 1:ntrain, ntrain+1:ntrain+ntest, ntrain+ntest+1:nrows

    if shuffle
        p = randperm(nrows)
        x = x[p,:]
        y = y[p,:]
    end
    
    x_train, x_test, x_val = x[train,:], x[test,:], x[val,:]
    y_train, y_test, y_val = y[train,:], y[test,:], y[val,:]

    (x_train, y_train), (x_test, y_test), (x_val, y_val)
end

## DATA LOADER
function data_loader(data, batch_size)
    # Calculate the total number of batches
    num_batches = Int(floor(length(data) / batch_size))
    # Partition data into batches
    batches = [ data[ min((i-1)*batch_size+1, num_batches-1):min(i*batch_size, end), : ] for i in 1:num_batches ]
    
    batches
end

end # END
