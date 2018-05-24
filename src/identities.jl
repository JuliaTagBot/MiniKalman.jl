# FillArrays.jl has `Zero` and `Eye`, which are good, but:
#  - vec + Zero(2) creates a new `vec` instead of returning `vec`, which is wasteful
#  - They include type info and length, which are kinda annoying to specify.
#    I just want a clean algebraic object.

struct IdentityMat end

Base.:*(::IdentityMat, x) = x
Base.:*(x, ::IdentityMat) = x
Base.transpose(::IdentityMat) = IdentityMat()

struct ZeroMat end
Base.:+(x, ::ZeroMat) = x
Base.:+(::ZeroMat, x) = x
