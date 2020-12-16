using FastL2LiR
using Test

@testset "FastL2LiR.jl" begin

    # Array size tests

    X = rand(10, 200)
    Y = rand(10, 100)
    Xp = rand(8, 200)

    model = fit(X, Y, 100.0)
    pred = predict(model, Xp)

    @test size(model.W) == (200, 100)
    @test size(model.b) == (1, 100)
    @test size(pred) == (8, 100)

    model = fit(X, Y, 100.0, 100)

    @test size(model.W) == (200, 100)
    @test size(model.b) == (1, 100)

    X = rand(10, 200)
    Y = rand(10, 5, 4, 3)
    Xp = rand(8, 200)

    model = fit(X, Y, 100.0)
    pred = predict(model, Xp)

    @test size(model.W) == (200, 5, 4, 3)
    @test size(model.b) == (1, 5, 4, 3)
    @test size(pred) == (8, 5, 4, 3)

    model = fit(X, Y, 100.0, 100)

    @test size(model.W) == (200, 5, 4, 3)
    @test size(model.b) == (1, 5, 4, 3)
end
