
export mse, mae

mse(y_pred, y) = mean((y_pred - y).*(y_pred - y))
mae(y_pred, y) = mean(abs.(y_pred - y))
