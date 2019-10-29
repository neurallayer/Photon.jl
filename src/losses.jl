
export mse, mae

mae(y_pred, y) = mean(abs.(mat(y_pred) .- mat(y)))

mse(y_pred, y) = mean((mat(y_pred) .- mat(y)).^2)
