# depend.jl

using Images, ImageFiltering, ImageTransformations
using Interpolations

# Define helper functions


# Load float image
function loadFloatImage(path)
    return float(channelview(load(path)))
    # For a NxM image, this returns a 3xNxM Array.
    # To move the color channel to the last dimension:
    # ch2 = cat(3,ch[1,:,:], ch[2,:,:], ch[3,:,:])
    # This reshaping might not be necessary
    # look into permutedims, which makes a new array and is better for adjacent
    #      pixels within a color channel
    # or look into permuteddimsview, which shares memory, and is better
    #      for accessing across color for a single pixel
end

# Load bit (binary) image
function loadBitImage(path)
    return (Images.Gray.(load(path)) .> 0.5)
end

function resizeFloatImage(img, n, m)
    # We assume the color channel is the first dimension of image
    i_nch, i_n, i_m  = size(img)

    # Shrinking? Let's filter first
    if i_n > n || i_m > m
        σ = map((o,n)->0.75*o/n, size(img), sz)
        kern = KernelFactors.gaussian(σ)   # from ImageFiltering
        imgr = imresize(imfilter(img, kern, NA()), (i_nch,n,m))
    elseif i_n < n || i_m < m
        imgr = imresize(img, (i_nch, n, m))
    elseif i_n == n && i_m == m
        imgr = img
    end

    return imgr
end

meshgrid(v::AbstractVector) = meshgrid(v, v)

function meshgrid(vx::AbstractVector{T}, vy::AbstractVector{T}) where T
    m, n = length(vy), length(vx)
    vx = reshape(vx, 1, n)
    vy = reshape(vy, m, 1)
    (repmat(vx, m, 1), repmat(vy, 1, n))
end

function meshgrid(vx::AbstractVector{T}, vy::AbstractVector{T},
                  vz::AbstractVector{T}) where T
    m, n, o = length(vy), length(vx), length(vz)
    vx = reshape(vx, 1, n, 1)
    vy = reshape(vy, m, 1, 1)
    vz = reshape(vz, 1, 1, o)
    om = ones(Int, m)
    on = ones(Int, n)
    oo = ones(Int, o)
    (vx[om, :, oo], vy[:, on, oo], vz[om, on, :])
end

export meshgrid

function imWarp(img, dx, dy)
    # use dx and dy as offset
    nchannels, imheight, imwidth = size(img)
    height, width = size(dx)

    xx, yy = meshgrid(1:imwidth, 1:imheight)
    XX, YY = meshgrid(1:width, 1:height)

    # using offset
    XX = XX + dx
    YY = YY + dy

    #mask = XX < 1 | XX > imwidth | YY < 1 | YY > imheight
    XX = minimum([maximum([XX, 1]), imwidth])
    YY = minimum([maximum([YY, 1]), imheight])

    for i = 1:nchannels
        itp = interpolate(img, BSpline(Linear()), OnGrid())
        imw[i,:,:] = itp[XX, YY]
    end

return imw

end

export imWarp
