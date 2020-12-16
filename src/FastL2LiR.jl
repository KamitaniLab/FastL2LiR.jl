"""
Fast L2-regularized Linear Regression

Kei Majima developed the original algorithm.
Soma Nonaka implemented the initial Julia version.
Shuntaro C. Aoki modified and refactored the original implementation.
"""

module FastL2LiR

export fit, predict

using Statistics
using LinearAlgebra

struct FastL2LiRModel{T<:Real}
    W::Array{T}
    b::Array{T}
end

function fit(X::Array{T, 2}, Y::Array{T}, alpha::T)::FastL2LiRModel where {T<:Real}
    return _fit(X, Y, alpha, false, 0)
end

function fit(X::Array{T, 2}, Y::Array{T}, alpha::T, num_features::Integer)::FastL2LiRModel where {T<:Real}
    return _fit(X, Y, alpha, true, num_features)
end

function _fit(X::Array{T, 2}, Y::Array{T}, alpha::T, feature_selection::Bool, num_features::Integer)::FastL2LiRModel where {T<:Real}
    """
    Fit L2-regularized linear regression model.
    """

    reshape_y::Bool = false
    if ndims(Y) > 2
        reshape_y = true
        y_shape = size(Y)[2:end]
        Y = reshape(Y, size(Y, 1), :)
    end

    n_sample = size(X, 1)
    n_feat = size(X, 2)
    n_trg = size(Y, 2)

    if n_feat < num_features
        feature_selection = false
    end

    # Append bias term in X
    Xb = [X ones(T, n_sample)]

    if feature_selection
        # With feature selection
        W = zeros(T, n_feat, n_trg)
        b = zeros(T, n_trg)

        C = cor(X, Y, dims=1)
        C = @. abs(C)
        C[C .== NaN] .= 0

        W0 = Xb' * Xb + alpha * I(n_feat + 1)
        W1 = Xb' * Y

        # @inbounds causes seg fault with MKL-linked Julia
        for i = 1:n_trg
            C0 = C[:, i]
            feature_index = sortperm(C0, rev=true)[1:num_features]
            push!(feature_index, size(Xb, 2))  # バイアス項のインデックスを追加する

            try
                Wb = W0[feature_index, feature_index] \ W1[feature_index, i]
            catch
                println("Singular matrix: use QR decomposition")
                Q, R = qr(W0)
                Wb = inv(R) * (Q' * W1[feature_index, i])
            end

            W[feature_index[1:end - 1], i] = Wb[1:end - 1, 1]
            b[i] = Wb[end, 1]
        end
    else
        # Without feature selection
        Wb = (Xb' * Xb + alpha * I(n_feat + 1)) \ (Xb' * Y)
        W = Wb[1:end - 1, :]
        b = Wb[end, :]
    end

    # Rerurns b as a row vector (compat. for PyFastL2LiR)
    b = collect(b')

    if reshape_y
        W = reshape(W, n_feat, y_shape...)
        b = reshape(b, 1, y_shape...)
    end

    return FastL2LiRModel(W, b)
end

function predict(model::FastL2LiRModel, X::Array{T, 2})::Array{T} where {T<:Real}
    """
    Predict y with given x and L2-regularized linear regression model.
    """

    reshape_y::Bool = false
    if ndims(model.W) > 2
        reshape_y = true
        w_shape = size(model.W)[2:end]
        W = reshape(model.W, size(model.W, 1), :)
        b = reshape(model.b, 1, :)
    else
        W = model.W
        b = model.b
    end

    Y_pred = X * W + repeat(b, size(X, 1), 1)

    if reshape_y
        Y_pred = reshape(Y_pred, size(Y_pred, 1), w_shape...)
    end

    return Y_pred
end

end # module FastL2LiR
