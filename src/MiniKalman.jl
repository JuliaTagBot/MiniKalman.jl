module MiniKalman

using GaussianDistributions, FillArrays
using GaussianDistributions: dim, logpdf
using StaticArrays
using Statistics, Random, LinearAlgebra

export kfilter, kalman_filter, white_noise1, white_noise2,
    kalman_smoother, kalman_sample, no_noise, log_likelihood, cumulative_log_likelihood

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

predicted_state(state_prior::Gaussian, transition_mat, transition_noise::Gaussian) =
    # Helper. Returning a tuple is more convenient than a Gaussian
    (transition_mat * mean(state_prior) + mean(transition_noise),
     transition_mat * cov(state_prior) * transition_mat' + cov(transition_noise))

""" Perform one step of Kalman filtering, for online use. We assume equations:

```julia
current_state = transition_mat * state_prior + transition_noise
observation = observation_mat * current_state + observation_noise
```

and return `current_state::Gaussian`, which is the posterior `P(state|observation)`.

To add an "input" term, pass it as `transition_noise = Gaussian(input_term, noise_cov)`.
"""
function kfilter(state_prior::Gaussian, transition_mat, transition_noise::Gaussian,
                 observation, observation_mat, observation_noise::Gaussian)
    # Following Machine Learning, A probabilistic perspective, by Kevin Murphy, 18.3.1.2,
    # with the exception that I'm setting the forcing term to 0, but allowing noise terms
    # with means (without loss of generality)
    # TODO: use keyword arguments on 0.7

    Du, R = parameters(observation_noise)     # Du := Dₜuₜ
    A = transition_mat
    C = observation_mat
    y = observation

    # Prediction step (μ_(t|t-1), Σ_(t|t-1))
    μ, Σ = predicted_state(state_prior, transition_mat, transition_noise) 
    ŷ = C * μ + Du               # Murphy forgot the Du term in his book
    S = C * Σ * C' + R
    predicted_obs = Gaussian(ŷ, S)
    
    # Filter
    K = Σ * C' / S         # Kalman gain matrix
    r = y - ŷ
    filtered_state = Gaussian(μ + K*r,
                              (I - K*C) * Σ)
    ll = logpdf(predicted_obs, y)  # log-likelihood
    return (filtered_state, ll, predicted_obs)
end

""" `make_full` is a helper """
make_full(v) = v
make_full(g::Gaussian) = Gaussian(make_full(mean(g)), make_full(cov(g)))
make_full(d::Diagonal) = convert(Matrix, d)
# See StaticArrays#468. We should probably use Diagonal() in 0.7
make_full(d::SDiagonal{1, Float64}) = @SMatrix [d[1,1]]
make_full(d::SDiagonal{2, Float64}) = @SMatrix [d[1,1] 0.0; 0.0 d[2,2]]
make_full(d::SDiagonal{N}) where N =  # this version allocates on 0.6!!!
    convert(SMatrix{N,N}, Diagonal(diag(d)))

kalman_filtered(args...; kwargs...) = kalman_filter(args...; kwargs...)[1]  # convenience

""" Convenience; returns a vector of the total likelihood up to each step. """
cumulative_log_likelihood(args...; kwargs...) =
    cumsum(kalman_filter(args...; kwargs...)[2])
""" Compute the smoothed belief state at step `t`, given the `t+1`'th smoothed belief
state. """
function ksmoother(filtered_state::Gaussian, next_smoothed_state::Gaussian,
                   next_transition_mat, next_transition_noise::Gaussian)
    # Notation:
    #    ₜ₁ means t+1
    #    Xₜₜ means (Xₜ|data up to t)
    #    T means "all data past, present and future"

    # Deconstruct arguments
    Aₜ₁ = next_transition_mat
    μₜₜ, Σₜₜ = parameters(filtered_state)
    μₜ₁T, Σₜ₁T = parameters(next_smoothed_state)

    # Prediction step
    μₜ₁ₜ, Σₜ₁ₜ =
        predicted_state(filtered_state, next_transition_mat, next_transition_noise)

    # Smoothed state
    # I don't like to use the inverse (the other equation is in theory more accurate),
    # but until StaticArrays#73... Note that `lu(::StaticArray)` is defined and might
    # be used, and Σ is positive definite, so there might be faster algorithms.
    J = Σₜₜ * Aₜ₁' * inv(Σₜ₁ₜ)       # backwards Kalman gain matrix
    #J = Σₜₜ * Aₜ₁' / Σₜ₁ₜ       # backwards Kalman gain matrix
    return Gaussian(μₜₜ + J * (μₜ₁T - μₜ₁ₜ),
                    Σₜₜ + J * (Σₜ₁T - Σₜ₁ₜ) * J')
end

include("kalman_models.jl")

end  # module
