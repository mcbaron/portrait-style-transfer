# style-transfer-survey

## Survey of Image Style Transfer Methodologies


The proposed project is an examination of different methods for image style transfer, from a linear algebraic construction through automatic and convolutional neural networks. Hopefully to understand what patterns and features a machine can learn that underlie human image perception.


## Setup
With a working installation of Julia, handily provided via the [JuliaPro project](https://juliacomputing.com/products/juliapro.html) for ease of installation.

Update any packages that need updating, `Pkg.update()` at the REPL. This project requires adding the following Julia packages:
- Images.jl
- ImageSegmentation
- Colors.jl
- Cxx.jl (requires cmake 3.4.3+)
- MATLAB.jl

We're going to use the [dlib](http://dlib.net/compile.html) C++ library to do facial landmark detection on our input and reference images. We won't need the entire library, because we're going to bootstrap the landmark detection code from Julia.



## References
- JuliaPro Project
- dlib
-
