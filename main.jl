#cd("/home/mcbaron/SpiderOak Hive/School/Masters 17-18/18-794/Project/style-transfer-survey")

using Images, FileIO, ImageView, Colors, DataFrames
# Include helper functions
include("depend.jl")

in_path = "images/input/DSC_3305_Lisa_Passport_scale.png"
ref_path = "images/reference/16.png"

landmarked = true
masks_exist = true
mask_in_path = "images/masks/DSC_3305_Lisa_Passport_scale.png"
mask_ref_path = "images/masks/16.png"

bg_ref_path = "images/bgs/16.png"

# TODO: Check if resize needed. Use resizeFloatImage
im_in = loadFloatImage(in_path)
im_ref = loadFloatImage(ref_path)

if masks_exist
    mask_in = loadBitImage(mask_in_path)
    mask_ref = loadBitImage(mask_ref_path)
else
    # For both the input and reference images:
    # - Need to mask out face from background -> save out as foreground image
    # - Mask out background from face,
    #   inpaint it to develop smooth canvas for replacement -> save out as background image
    # - Need the mask that distinguishes foreground + background -> save out as mask image
    # Hopefully this can be done with
    # ImageSegmentation.jl: https://juliaimages.github.io/latest/imagesegmentation.html
end

if landmarked
    lm_in = Array{Float16}(readtable("images/landmarks/DSC_3305_Lisa_Passport_scale.lm", separator=','))
    lm_ref = Array{Float16}(readtable("images/landmarks/16.lm", separator=','))
else
    # - Detect facial landmarks automatically for both input and reference image
    #        -> using dlib and Cxx.jl
end

# ----//---- Dense Match ----//----
# Either by morphing or Procrustes, or both?
# Refine with SIFT Flow? - Call out to MATLAB for this.
# Threshold and warp image + mask

# We're aiming to warp the reference image to fit the input image.
im_refwarp, vx, vy = imageMorph(im_ref, im_in, lm_ref, lm_in)
if show_match
    ImageView(im_refwarp + im_in)
end


# ----//---- Local Power Matching ----//----

# ----//---- Matting (background transfer)  ----//----

# ----//---- Eye highlight transfer ----//----

# Save out
