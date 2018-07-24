module MiniKalman

using GaussianDistributions, FillArrays
using GaussianDistributions: dim, logpdf
using StaticArrays

export kfilter, kalman_filter, white_noise, white_noise1, white_noise2,
    kalman_smoother, kalman_sample, no_noise

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

struct Zero end
Base.:+(x, ::Zero) = x
Base.:+(::Zero, x) = x
Base.:*(x, ::Zero) = zero(x)
Base.:*(::Zero, x) = zero(x)
Base.transpose(z::Zero) = z

################################################################################

no_noise() = Gaussian(Zero(), Zero())
white_noise2(sigma2s...) =
    Gaussian(zero(SVector{length(sigma2s), Float64}), SDiagonal(sigma2s))
# fast special-cases
white_noise2(a) = Gaussian(SVector(0.0), SMatrix{1,1}(a)) 
white_noise2(a, b) = Gaussian(SVector(0.0, 0.0), SDiagonal(a, b))
# TODO: eventually have white_noise = white_noise1 and maybe stop exporting white_noise2
# since it's counter-intuitive, and deprecate white_noise1.
# We've been white_noise-free since June 7th.
white_noise1(args...) = white_noise2((args.^2)...)
white_noise(args...) = white_noise1(args...)

parameters(g::Gaussian) = (mean(g), cov(g))   # convenience

predicted_state(state_prior::Gaussian, transition_mat, transition_noise::Gaussian) =
    # Helper. Returning a tuple is more convenient than a Gaussian
    (transition_mat * mean(state_prior) + mean(transition_noise),
     transition_mat * cov(state_prior) * transition_mat' + cov(transition_noise))

# function Base.lufact(m::SMatrix)
#     # Necessary for kalman_smoother until StaticArrays#73 (... I guess?)
#     return lufact(convert(Matrix, m))
#     #return Base.LinAlg.LU(convert(typeof(m), lu.factors), lu.ipiv, lu.info)
# end

# Type piracy! This (mathematically correct) definition improves filtering speed by 3X!
# I believe that it could also easily support a diagonal A, but that's not useful for us.
# See definitions in Base.
Base.:\(A::StaticArrays.SArray{Tuple{1,1},<:Any,2,1},
        B::StaticArrays.SArray) = B ./ A[1]

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

kalman_filter(initial_state_prior::Gaussian, observations::AbstractVector,
              observation_noises::AbstractVector{<:Gaussian};
              _N=length(observations), # "hidden" kwargs to help create defaults
              transition_mats::AbstractVector=Fill(Identity(), _N),
              transition_noises::AbstractVector{<:Gaussian}=
              Fill(no_noise(), _N),
              observation_mats::AbstractVector=Fill(Identity(), _N)) =
    kalman_filter(initial_state_prior, observations,
                  observation_noises, transition_mats, transition_noises,
                  observation_mats)

""" `make_full` is a helper """
make_full(v) = v
make_full(g::Gaussian) = Gaussian(make_full(mean(g)), make_full(cov(g)))
make_full(d::Diagonal) = convert(Matrix, d)
# See StaticArrays#468. We should probably use Diagonal() in 0.7
make_full(d::SDiagonal{1, Float64}) = @SMatrix [d[1,1]]
make_full(d::SDiagonal{2, Float64}) = @SMatrix [d[1,1] 0.0; 0.0 d[2,2]]
make_full(d::SDiagonal{N}) where N =  # this version allocates on 0.6!!!
    convert(SMatrix{N,N}, Diagonal(diag(d)))

# I split off the non-kwarg version mostly for `@code_warntype` ease. Revisit in 0.7?
# It turned out to have a negligible impact on performance anyway. The bottle-neck
# was elsewhere. TODO: merge them together again?
function kalman_filter(initial_state_prior::Gaussian, observations::AbstractVector,
                       observation_noises::AbstractVector{<:Gaussian},
                       transition_mats::AbstractVector,
                       transition_noises::AbstractVector{<:Gaussian},
                       observation_mats::AbstractVector)
    @assert(length(observations) == length(transition_mats) ==
            length(transition_noises) == length(observation_mats) ==
            length(observation_noises),
            "All passed vectors should be of the same length")
    state = make_full(initial_state_prior)  # we need make_full to that the state does

    # not change type during iteration
    # For type stability, we fake-run it. It's rather lame. Ideally, we'd build the
    # output type from the input types
    _, _, dum_predictive =
        kfilter(initial_state_prior, transition_mats[1], transition_noises[1],
                observations[1], observation_mats[1], observation_noises[1])
    P = typeof(dum_predictive)
    T = typeof(state)
    filtered_states = Vector{T}(length(observations))
    predicted_obs = Vector{P}(length(observations))
    lls = Vector{Float64}(length(observations))

    for t in 1:length(observations)
        state, lls[t], predicted_obs[t] =
            kfilter(state, transition_mats[t], transition_noises[t],
                    observations[t], observation_mats[t], observation_noises[t])
        filtered_states[t] = state::T
    end
    return filtered_states, lls, predicted_obs
end

function log_likelihood(initial_state_prior::Gaussian, observations::AbstractVector,
                        observation_noises::AbstractVector{<:Gaussian},
                        transition_mats::AbstractVector,
                        transition_noises::AbstractVector{<:Gaussian},
                        observation_mats::AbstractVector)
    # Specialized version that doesn't allocate at all. Useful for parameter optimization.
    ll_sum = 0.0
    state = make_full(initial_state_prior)
    for t in 1:length(observations)
        state, ll, predictive =
            kfilter(state, transition_mats[t], transition_noises[t],
                    observations[t], observation_mats[t], observation_noises[t])
        ll_sum += ll
    end
    return ll_sum
end    

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

function kalman_smoother(filtered_states::AbstractVector{<:Gaussian};
                         transition_mats::AbstractVector=
                             Fill(Identity(), length(filtered_states)),
                         transition_noises::AbstractVector{<:Gaussian}=
                             Fill(no_noise(),
                                  length(filtered_states)))
    smoothed_states = fill(filtered_states[end], length(filtered_states))
    for t in length(smoothed_states)-1:-1:1
        smoothed_states[t] =
              ksmoother(filtered_states[t], smoothed_states[t+1],
                        transition_mats[t+1], transition_noises[t+1])
    end
    return smoothed_states
end

################################################################################
# Sampling

# This is essentially the definition of sampling from a dirac delta. 
Base.rand(RNG, P::Gaussian{U, Zero}) where U = P.μ

""" Returns `(hidden_state::Vector, observations::Vector)` """
function kalman_sample(rng::AbstractRNG, initial_state,
                       observation_noises::AbstractVector{<:Gaussian};
                       _N=length(observation_noises), # to help create defaults
                       transition_mats::AbstractVector=Fill(Identity(), _N),
                       transition_noises::AbstractVector{<:Gaussian}=
                           Fill(no_noise(), _N),
                       observation_mats::AbstractVector=Fill(Identity(), _N))
    @assert(length(transition_mats) ==
            length(transition_noises) == length(observation_mats) ==
            length(observation_noises),
            "All passed vectors should be of the same length")
    result = accumulate((initial_state, nothing), 1:_N) do v, t
        state, _ = v
        next_state = transition_mats[t] * state + rand(rng, transition_noises[t])
        return (next_state,
                observation_mats[t] * next_state + rand(rng, observation_noises[t]))
    end
    return (map(first, result), map(last, result))
end


include("kalman_models.jl")

end  # module
