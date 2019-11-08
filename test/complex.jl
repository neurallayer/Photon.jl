module ComplexTests

using Photon, Test


"""
Model that contains two sequential submodels in order to
test a two-input, two-output scenario. The model itself
is not very meaningful ;)
"""
struct MyModel <: Photon.Layer
    a
    b

    function MyModel()
        a = Sequential(Dense(128), Dense(20))
        b = Sequential(Conv2D(16,3), Dense(10))
        new(a, b)
    end
end

function (model::MyModel)(X)
    X1, X2 = X
    model.a(X1), model.b(X2)
end

"""
Loss function that takes the two outputs and returns a single
loss value.
"""
function myloss(y_pred, y)
    p1, p2 = y_pred
    y1, y2 = y
    (mse(p1, y1) + mae(p2, y2))/2
end

"""
Test a model that has multiple input and outputs. The default convertor
(autoConvertor) will take care of handling this data while preserving this
structure.
"""
function multi_input_output()

    model = MyModel()

    workout = Workout(model, myloss)
    dtype = getContext().dtype

    # Two inputs
    X = [(randn(dtype,100,4),randn(dtype,50,50,3,4)) for i in 1:10]

    # Two outputs
    Y = [(randn(dtype,20,4), randn(dtype,10,4)) for i in 1:10]

    fit!(workout, zip(X,Y), epochs=2)

    @test hasmetric(workout, :loss)

end


@testset "Complex" begin
    multi_input_output()
end

end
