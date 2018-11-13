################################################################################
# Algebraic identities
#
# FillArrays.jl has `Zero` and `Eye`, which are good, but:
#  - vec + Zero(2) creates a new `vec` instead of returning `vec`, which is wasteful
#  - They include type info and length, which are annoying to specify.
#    We just need a clean algebraic object.

struct Identity end
Base.:*(::Identity, x) = x
Base.:*(x, ::Identity) = x
Base.:*(g::Gaussian, ::Identity) = g  # disambiguation
Base.:*(::Identity, g::Gaussian) = g  # disambiguation
Base.transpose(::Identity) = Identity()
Base.adjoint(::Identity) = Identity()

struct Zero end
Base.:+(x, ::Zero) = x
Base.:+(::Zero, x) = x
Base.:*(x, ::Zero) = zero(x)
Base.:*(::Zero, x) = zero(x)
Base.transpose(z::Zero) = z
Base.adjoint(z::Zero) = z

################################################################################

""" `make_full` is a helper """
make_full(v) = v
make_full(g::Gaussian) = Gaussian(make_full(mean(g)), make_full(cov(g)))
make_full(d::Diagonal) = convert(Matrix, d)
# See StaticArrays#468. We should probably use Diagonal() in 0.7
make_full(d::SDiagonal{1, Float64}) = @SMatrix [d[1,1]]
make_full(d::SDiagonal{2, Float64}) = @SMatrix [d[1,1] 0.0; 0.0 d[2,2]]
make_full(d::SDiagonal{N}) where N =  # this version allocates on 0.6!!!
    convert(SMatrix{N,N}, Diagonal(diag(d)))

################################################################################

no_noise() = Gaussian(Zero(), Zero())
white_noise2(sigma2s...) =
    Gaussian(zero(SVector{length(sigma2s), typeof(zero(sqrt(sigma2s[1])))}),
             SDiagonal(sigma2s))
# fast special-cases
white_noise2(a) =
    # sqrt(zero(a)) is non-differentiable, but zero(sqrt(a)) is.
    Gaussian(SVector(zero(sqrt(a))), SMatrix{1,1}(a))
white_noise2(a, b) = Gaussian(SVector(zero(sqrt(a)), zero(sqrt(b))), SDiagonal(a, b))
white_noise1(args...) = white_noise2((args.^2)...)

################################################################################

Random.rand(RNG::AbstractRNG, g::Gaussian{<:SVector{1}}) =  # type piracy!
    # otherwise calls Cholesky, which fails with unitful Gaussians
    SVector(rand(RNG, Gaussian(mean(g)[1], cov(g)[1])))

parameters(g::Gaussian) = (mean(g), cov(g))   # convenience

# function Base.lufact(m::SMatrix)
#     # Necessary for kalman_smoother until StaticArrays#73 (... I guess?)
#     return lufact(convert(Matrix, m))
#     #return Base.LinAlg.LU(convert(typeof(m), lu.factors), lu.ipiv, lu.info)
# end

# Type piracy! This (mathematically correct) definition improves filtering speed by 3X!
# I believe that it could also easily support a diagonal A, too.
# See definitions in Base.
Base.:\(A::StaticArrays.SArray{Tuple{1,1},<:Any,2,1},
        B::StaticArrays.SArray) = B ./ A[1]

################################################################################
# Marginal variance and standard-deviation

marginal_var(g::Gaussian) = diag(cov(g))
marginal_var(g::Gaussian, i::Int) = diag(cov(g))[i]
marginal_std(args...) = sqrt(marginal_var(args...))
marginal(g::Gaussian, i::Int) = Gaussian(mean(g)[i], marginal_var(g, i))
is_marginal(g::Gaussian) = dim(g) == 1  # a bit wonky
