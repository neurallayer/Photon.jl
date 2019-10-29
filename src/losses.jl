
export mse, mae

mae(y_pred, y) = mean(abs.(Knet.mat(y_pred) .- Knet.mat(y)))

mse(y_pred, y) = mean((Knet.mat(y_pred) .- Knet.mat(y)).^2)
