using Images, Colors, Interpolations
using Polynomials
using PyCall
using Interpolations
using Convex
using JSON
using IterativeSolvers
using Interact # for displaying signals of images
using FileIO: @format_str, File, filename, add_format, stream

using Plots: plot, scatter, plot!, scatter!, heatmap, heatmap!, @recipe, plotlyjs

function tsvd(A, k = min(min(size(A)...), 6))
    m, n = size(A)
    minmn = min(m, n)

    if k == minmn               # All values               : QR algorithm (density if necessary)
        return svd(full(A))
    elseif k > 0
        if k > sqrt(minmn) + 1  # More than sqrt(n) values : QR algorithm (density if necessary)
            U, s, V = svd(full(A))
            return U[:,1:k], s[1:k], V[:,1:k]
        else                    # less than sqrt(n) values : Lanczos algorithm
            out = svds(A, nsv = k)
            return out[1][:U], out[1][:S], VERSION < v"0.6.0-dev.2026" ? out[1][:V]' : out[1][:V]
        end
    else
        return Array{eltype(A)}(m, 0), Array{eltype(A)}(0), Array{eltype(A)}(n, 0)
    end
end

function spy(A::Matrix;
        xaxis = 1:size(A, 2), yaxis = 1:size(A, 1),
        kw_args...
    )
    heatmap(flipdim(A, 1), xaxis = xaxis, yaxis = yaxis; kw_args...)
end

# Apply mask
function applyMask(Y, M, color)
    m, n = size(Y)
    Y = repmat(Y[:], 1, 3)
    Y[M[:], :] = repmat(color[:]', countnz(M), 1)
    return reshape(Y, m, n, 3)
end

# Load float image
function loadFloatImage(path)
    return Float64.(Images.Gray.(load(path)))
end

# Load bit (binary) image
function loadBitImage(path)
    return (Images.Gray.(load(path)) .> 0.5)
end

function imshow(X; kw_args...)
    return plot(
        X,
        st = :heatmap,
        color = :grays,
        yflip = true,
        aspect_ratio = :equal,
        border = false,
        ticks = [];
        kw_args...,
    )
end
