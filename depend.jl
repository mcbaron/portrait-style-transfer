# depend.jl

using Images, ImageFiltering, ImageTransformations
using Interpolations, ColorVectorSpace

# Define helper functions


# Load float image
function loadFloatImage(path)
  # return float(channelview(load(path)))
  return load(path)
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
  om = ones(Float32, m)
  on = ones(Float32, n)
  oo = ones(Float32, o)
  (vx[om, :, oo], vy[:, on, oo], vz[om, on, :])
end

export meshgrid

function imWarp(img, dx, dy)
    itp = interpolate(img, BSpline(Linear()), OnGrid())
    inds = indices(img)
    rng = extrema.(inds)
    imw = similar(img, eltype(itp))
    for I in CartesianRange(inds)
        dxi, dyi = dx[I], dy[I]
        y, x = clamp(I[1]+dyi, rng[1]...), clamp(I[2]+dxi, rng[2]...)
        imw[I] = itp[y, x]
    end
    return imw
end

export imWarp

function channelHistogramTransfer(dest_chan, source_chan)
    sorted_source = sort(source_chan[:])
    dest_idx = sortperm(dest_chan[:])
    output_chan = zeros(size(dest_chan))
    output_chan[dest_idx] = sorted_source

    return output_chan
end

export channelHistogramTransfer


function extract_L(image_in)
  return [x.l for x in image_in]
end
export extract_L


"""
```
pyramid = laplacian_pyramid(img, n_scales, sigma)
```

Returns a  laplacian pyramid of scales `n_scales`, each filtered by
a gaussian kernel of increasing σ.

"""
function laplacian_pyramid(img::AbstractArray{T,N}, n_scales::Int) where {T,N}

  # To guarantee inferability, we make sure that we do at least one
  # round of smoothing
  sigma = 2
  kerng = KernelFactors.IIRGaussian(sigma)
  kern = ntuple(d->kerng, Val{N})
  img_smoothed = imfilter(img, kern, Images.NA())
  pyramid = typeof(img_smoothed)[img - img_smoothed]
  prev = img_smoothed
  for i in 2:n_scales
    sigma = 2^i
    kerng = KernelFactors.IIRGaussian(sigma)
    kern = ntuple(d->kerng, Val{N})
    img_smoothed = imfilter(prev, kern, Images.NA())
    img_diff = prev - img_smoothed

    push!(pyramid, img_diff)
    prev = img_smoothed
  end
  push!(pyramid, img_smoothed) # attach the residual as the last layer
  return pyramid
end

export laplacian_pyramid

function reconstruct_laplacian_pyramid(pyramid)
  n_scales = length(pyramid)
  img_out = pyramid[end]
  for i = n_scales-1:-1:1
    img_next = pyramid[i]
    img_out += img_next
  end
  return img_out
end

export reconstruct_laplacian_pyramid

function pyramid_scale(img, downsample)
    sz_next = map(s->ceil(Int, s/downsample), size(img))
    imresize(img, sz_next)
end
