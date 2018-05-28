module MiniKalman

using GaussianDistributions, FillArrays
using GaussianDistributions: dim, logpdf
using StaticArrays

export kfilter, kalman_filter, white_noise, kalman_smoother, kalman_sample, no_noise

################################################################################
# Algebraic identities
#
# FillArrays.jl has `Zero` and `Eye`, which are good, but:
#  - vec + Zero(2) creates a new `vec` instead of returning `vec`, which is wasteful
#  - They include type info and length, which are kinda annoying to specify.
#    I just want a clean algebraic object.

struct Identity end
Base.:*(::Identity, x) = x
Base.:*(x, ::Identity) = x
Base.:*(g::Gaussian, ::Identity) = g  # disambiguate
Base.:*(::Identity, g::Gaussian) = g  # disambiguate
Base.transpose(::Identity) = Identity()

struct Zero end
Base.:+(x, ::Zero) = x
Base.:+(::Zero, x) = x

################################################################################

parameters(g::Gaussian) = (mean(g), cov(g))

predicted_state(state_prior::Gaussian, transition_mat, transition_noise::Gaussian) =
    # Helper. Returning a tuple is more convenient than a Gaussian
    (transition_mat * mean(state_prior) + mean(transition_noise),
     transition_mat * cov(state_prior) * transition_mat' + cov(transition_noise))

function Base.lufact(m::SMatrix)
    # Necessary for kalman_smoother until StaticArrays#73 (... I guess?)
    return lufact(convert(Matrix, m))
    #return Base.LinAlg.LU(convert(typeof(m), lu.factors), lu.ipiv, lu.info)
end

# Type piracy! This definition improves filtering speed by 3X!
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
    
    # Filter
    S = C * Σ * C' + R
    K = Σ * C' / S         # Kalman gain matrix
    ŷ = C * μ + Du
    r = y - ŷ
    filtered_state = Gaussian(μ + K*r,
                              (I - K*C) * Σ)
    ll = logpdf(Gaussian(C * μ, S), y)  # log-likelihood
    return (filtered_state, ll)
end

no_noise() = Gaussian(Zero(), Zero())
white_noise(vals...) = Gaussian(SVector(ntuple(_->0.0, length(vals))), SDiagonal(vals))

kalman_filter(initial_state_prior::Gaussian, observations::AbstractVector,
              observation_noises::AbstractVector{<:Gaussian};
              _N=length(observations), # "hidden" kwargs to help create defaults
              transition_mats::AbstractVector=Fill(Identity(), _N),
              transition_noises::AbstractVector{<:Gaussian}=
              Fill(no_noise(), _N),
              # This default only makes sense if `d₂==d`
              observation_mats::AbstractVector=Fill(Identity(), _N)) =
    kalman_filter(initial_state_prior, observations,
                  observation_noises, transition_mats, transition_noises,
                  observation_mats)

# I split off the non-kwarg version mostly for `@code_warntype` ease. Revisit in 0.7?
# It turned out to have a negligible impact on performance anyway. The bottle-neck
# was elsewhere.
function kalman_filter(initial_state_prior::Gaussian, observations::AbstractVector,
                       observation_noises::AbstractVector{<:Gaussian},
                       transition_mats::AbstractVector,
                       transition_noises::AbstractVector{<:Gaussian},
                       observation_mats::AbstractVector)
    @assert(length(observations) == length(transition_mats) ==
            length(transition_noises) == length(observation_mats) ==
            length(observation_noises),
            "All passed vectors should be of the same length")
    # For type stability
    T = typeof(kfilter(initial_state_prior, transition_mats[1], transition_noises[1],
                       observations[1], observation_mats[1], observation_noises[1])[1])
    filtered_states = Vector{T}(length(observations))

    state = convert(T, initial_state_prior)
    ll_sum = 0.0
    for t in 1:length(observations)
        (state, ll) =
            kfilter(state, transition_mats[t], transition_noises[t],
                    observations[t], observation_mats[t], observation_noises[t])
        filtered_states[t] = state
        ll_sum += ll
    end
    return filtered_states, ll_sum
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
    J = Σₜₜ * Aₜ₁' / Σₜ₁ₜ       # backwards Kalman gain matrix
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
                       # This default only makes sense if `d₂==d`
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
