
export mse, mae, bce_loss

mae(y_pred, y) = mean(abs.(Knet.mat(y_pred) .- Knet.mat(y)))

mse(y_pred, y) = mean((Knet.mat(y_pred) .- Knet.mat(y)).^2)


binarycrossentropy(ŷ, y) = -y*log(ŷ + ϵ) - (1 - y)*log(1 - ŷ + ϵ)


function bce_loss(y_pred, y)
    y_pred = Knet.mat(y_pred)
    y = Knet.mat(y_pred)
    # y_pred = Knet.sigm.(y_pred)
    loss = binarycrossentropy.(y_pred, y)
    mean(loss)
end
