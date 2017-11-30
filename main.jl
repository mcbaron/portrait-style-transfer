#cd("/home/mcbaron/SpiderOak Hive/School/Masters 17-18/18-794/Project/style-transfer-survey")

using Images, FileIO, ImageView, Colors, DataFrames
# Include helper functions
include("depend.jl")
include("imageMorph.jl")
include("pyramids.jl")
in_path = "images/input/DSC_3305_Lisa_Passport_scale.png"
ref_path = "images/reference/16.png"

landmarked = true
masks_exist = true
mask_in_path = "images/masks/DSC_3305_Lisa_Passport_scale.png"
mask_ref_path = "images/masks/16.png"

bg_ref_path = "images/bgs/16.png"

im_in = loadFloatImage(in_path)
im_ref = loadFloatImage(ref_path)
# Check if resize needed on the input image:
m, n = size(im_in)
if m != 1320 || n != 1000 # We're using 1320x1000 reference images
  im_in = resizeFloatImage(im_in, 1320, 1000)
end
# Check if reszie needed on the reference image:
m, n = size(im_ref)
if m != 1320 || n != 1000
  iref = resizeFloatImage(im_ref, 1320, 1000)
end



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

    # The following works decently well in the REPL:
    # using ImageSegmentation
    # g = sum(im_in, 1)[1,:,:];
    # seg = fast_scanning(g, .5)
    # segments = prune_segments(seg, i->(segment_pixel_count(seg,i)<1000), (i,j)->(-segment_pixel_count(seg,j)))
    # imshow(map(i->get_random_color(i), labels_map(segments)))

    # function get_random_color(seed)
    #     srand(seed)
    #     rand(RGB{N0f8})
    # end

end

if landmarked
    lm_in = Array{Float32}(readtable("images/landmarks/DSC_3305_Lisa_Passport_scale.lm", separator=',', header=false))
    lm_ref = Array{Float32}(readtable("images/landmarks/16.lm", separator=',', header=false))
else
    # - Detect facial landmarks automatically for both input and reference image
    #        -> using dlib and Cxx.jl
end

# ----//---- Dense Match ----//----
# Either by morphing or Procrustes, or both?
# Refine with SIFT Flow? - Call out to MATLAB for this.
# Threshold and warp image + mask

# We're aiming to warp the reference image to fit the input image.
show_match = true
im_refwarp, vx, vy = imageMorph(im_ref, im_in, lm_ref, lm_in)
if show_match
    imshow(im_refwarp + im_in)
end


# ----//---- Local Power Matching ----//----
e0 = 1e-4
gain_max = 2.8
gain_min = .9
hist_trans = false

# Transfer the background from the refernce image
im_in = mask_in.*im_in + (1-mask_in)*loadFloatImage(bg_ref_path)
# transform to Lab colorspace, why isn't this working?
im_in = cconvert(im_in, Lab)
im_refwarp = cconvert(im_refwarp, Lab)

pyr_in = ImagePyramid(im_in, LaplacianPyramid(), max_levels = 8)
pyr_ref = ImagePyramid(im_refwarp, LaplacianPyramid(), max_levels = 8)



# ----//---- Matting (background transfer)  ----//----

# ----//---- Eye highlight transfer ----//----

# Save out
