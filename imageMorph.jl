function imageMorph(image_in, image_dest, lm_in, lm_dest)
 # Inputs: image_in - the input image, to be warped/morphed to the destination image
 #         image_dest - the destination image.
 #         lm_in - the landmarks corresponding to the input image
 #         lm_dest - the landmakrs corresponding to the destination image
 #
 #
 # Returns: im_warped - the input image aligned to the destination image
 #          vx - the x dimension warping map
 #          vy - the y dimension warping map

nch, n, m = size(image_in)
xx, yy = meshgrid(1:m, 1:n)

# construct segments between each landmark point
segs_src = landmark2Seg(lm_in, n, m)
segs_dst = landmark2Seg(lm_dest, n, m)

xsum = zeros(size(xx))
ysum = zeros(size(yy))
wsum = zeros(size(xx))
for i = 1:size(segs_src,1)
    u, v = get_uv(xx, yy, segs_src[i,:])
    x, y = get_xy(u, v, segs_dst[i,:])
    weight = get_weight(xx, yy, segs_src[i,:])
    wsum += weight
    xsum += weight.*x
    ysum += weight.*y
end
x_norm = xsum ./ wsum
y_norm = yxum ./ wsum
vx = xx - x_norm
vy = yy - y_norm

vx[x_norm < 1] = 0
vx[x_norm > m] = 0
vy[y_norm < 1] = 0
vy[y_norm > n] = 0

im_warped = imWarp(image_dest, vx, vy)

return im_warped, vx, vy
end



function landmark2Seg(lmArray, height, width)
    # Take a landmark array and return an array of touples that represent the
    # start and end points of each segment

    # oracle knowledge is needed about what should and shouldn't be connected.
    idx = [1:16; 18:21; 23:26; 28:30; 32:35; 37:61; 61:65]
    idy = [2:17; 19:22; 24:27; 29:31; 33:36; 38:42; 37; 44:48; 43; 50:60; 49; 66; 62:66]

    nkeypoints = size(idx,1)
    segs = Array{Tuple}(nkeypoints+4, 2)
    
    for i = 1:nkeypoints
        segs[i,1] = Tuple(lmArray[idx[i],:])
        segs[i,2] = Tuple(lmArray[idy[i],:])
    end

    # Include the corners of the image
    segs[end-3,:] = [(1,1), (1,height)]
    segs[end-2,:] = [(1,height), (width, height)]
    segs[end-1,:] = [(width, height), (width, 1)]
    segs[end,:] = [(width, 1), (1,1)]

    return segs
end

function get_uv(x, y, line)
    p = line[1]
    q = line[2]

    pq = (q[1] - p[1], q[2] - p[2])
    len = sqrt(pq[1]^2 + pq[2]^2)

    u = ((x-p[1])*pq[1] + (y-p[2])*pq[2]) / len^2
    # v is perpendicular
    v = ((x-p[1])*-pq[2] + (y-p[2])*pq[1]) / len

    return u, v
end

function get_xy(u, v, line)
    p = line[1]
    q = line[2]

    pq = (q[1] - p[1], q[2] - p[2])
    len = sqrt(pq[1]^2 + pq[2]^2)

    x = p[1] + u*pq[1] + (v*-pq[2])/ len
    y = p[2] + u*pq[2] + (v*pq[1])/len

    return x,y
end

function get_weight(x, y, line)
    a=10
    b=1
    p=1
    u, v = get_uv(x, y, line)

    d1 = ((x-line[2][1]).^2 + (y-line[2][2]).^2 ).^ .5
    d2 = ((x-line[1][1]).^2 + (y-line[1][2]).^2 ).^ .5
    dv = abs(v)
    dv[u>1] = d1[u>1]
    dv[u<0] = d2[u<0]

    pq = (q[1] - p[1], q[2] - p[2])
    len = sqrt(pq[1]^2 + pq[2]^2)

    return (len^p./(a+d)).^b
end
