module KalmanCore

using GaussianDistributions, FillArrays
using GaussianDistributions: dim, logpdf
using StaticArrays

export kfilter, kalman_filter, white_noise, kalman_smoother

predicted_state(state_prior::Gaussian, transition_mat, transition_noise::Gaussian) =
    # Helper
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
    Du = mean(observation_noise)     # = D_t * u_t
    R = cov(observation_noise)
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

no_noise(d) = Gaussian(Zeros(d), Zeros(d, d))
white_noise(vals...) = Gaussian(Zeros(length(vals)), SDiagonal(vals...))

function kalman_filter(initial_state_prior::Gaussian, observations::AbstractVector,
                       observation_noises::AbstractVector{<:Gaussian};
                       # "hidden" kwargs to help create defaults
                       _d=dim(initial_state_prior), _N=length(observations),
                       _d₂=length(observations[1]),
                       transition_mats::AbstractVector=Fill(Eye(_d), _N),
                       transition_noises::AbstractVector{<:Gaussian}=
                           Fill(no_noise(_d), _N),
                       # The default only makes sense if `d₂==d`
                       observation_mats::AbstractVector=Fill(Eye(_d₂, _d), _N))
    @assert(length(observations) == length(transition_mats) ==
            length(transition_noises) == length(observation_mats) ==
            length(observation_noises),
            "All passed vectors should be of the same length")
    state = initial_state_prior
    filtered_states = fill(initial_state_prior, _N)
    total_ll = 0.0   # log-likelihood
    for t in 1:length(observations)
        state, ll = kfilter(state, transition_mats[t], transition_noises[t],
                            observations[t], observation_mats[t], observation_noises[t])
        filtered_states[t] = state
        total_ll += ll
    end
    return (filtered_states, total_ll)
end

""" Compute the smoothed belief state at step `t`, given the `t+1`'th smoothed belief
state. """
function ksmoother(filtered_state::Gaussian, next_smoothed_state::Gaussian,
                   next_transition_mat::AbstractMatrix, next_transition_noise::Gaussian)
    # Notation:
    #    ₜ₁ means t+1
    #    Xₜₜ means (Xₜ|data up to t)
    #    T means "all data past, present and future"

    # Deconstruct arguments
    Aₜ₁ = next_transition_mat
    Buₜ₁ = mean(next_transition_noise)      # = B_t * u_t   (input/control)
    Qₜ₁ = cov(next_transition_noise)
    μₜₜ, Σₜₜ = mean(filtered_state), cov(filtered_state)
    μₜ₁T, Σₜ₁T = mean(next_smoothed_state), cov(next_smoothed_state)

    # Predicted state
    transitioned_state = Aₜ₁ * filtered_state + Buₜ₁
    μₜ₁ₜ = Aₜ₁ * mean(transitioned_state)         # = μ_(t|t-1)
    Σₜ₁ₜ = cov(transitioned_state) + Qₜ₁    # = Σ_(t+1|t)

    # Smoothed state
    J = Σₜₜ * Aₜ₁' / Σₜ₁ₜ       # backwards Kalman gain matrix
    return Gaussian(μₜₜ + J * (μₜ₁T - μₜ₁ₜ),
                    Σₜₜ + J * (Σₜ₁T - Σₜ₁ₜ) * J')
end

function kalman_smoother(filtered_states::AbstractVector{<:Gaussian};
                         transition_mats::AbstractVector=
                             Fill(Eye(dim(filtered_states[1])), length(filtered_states)),
                         transition_noises::AbstractVector{<:Gaussian}=
                             Fill(no_noise(dim(filtered_states[1])),
                                  length(filtered_states)))
    smoothed_states = fill(filtered_states[end], length(filtered_states))
    for t in length(smoothed_states)-1:-1:1
        smoothed_states[t] =
              ksmoother(filtered_states[t], smoothed_states[t+1],
                        transition_mats[t+1], transition_noises[t+1])
    end
    return smoothed_states
end


end  # module
