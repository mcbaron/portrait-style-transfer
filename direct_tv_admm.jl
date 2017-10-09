# Let's start with TV inpainting
#
cd("/home/mcbaron/SpiderOak Hive/School/Masters 17-18/18-794/Project/style-transfer-survey")
# Load helper functions
include("deps.jl")
include("tv_admm.jl")
using Images
using FileIO
using PyPlot
plt = PyPlot

inpath = "/images/input/in1.png"
maskpath = "./images/segmentation/in1.png"
tarpath = "./images/style/tar1.png"

M = loadBitImage(maskpath)
Xin = loadFloatImage(inpath)
# Xin = RGB{Float64}.(load(inpath))
Yin = loadFloatImage(tarpath)
# Yin = RGB{Float64}.(load(tarpath))

plt.imshow(M, vmin=0, vmax=1, cmap="Greys_r")
plt.axis("equal"); plt.axis("off")
plt.title("Mask");

μ = 1e-3
# ADMM parameter
# Any rho > 0 should work, but convergence rate may change
ρ = 1
# Number of iterations
nIters = 20

T, cost = tv_admm(Xin[1:393,:], Yin, .!M[1:393,:], μ, ρ, nIters)

plt.imshow(T[:,:,end], vmin=0, vmax=1)
save("testTransfer.jpg",map(clamp01nan, T[:,:,end]))

plt.imshow(Xin[1:393,:], vmin=0, vmax=1)
