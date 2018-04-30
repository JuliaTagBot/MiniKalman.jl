module KalmanCore

using GaussianDistributions, FillArrays
using GaussianDistributions: dim, logpdf
using StaticArrays

export kfilter, kalman_filter, white_noise, kalman_smoother

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
    # with means. This is mathematically equivalent.
    # TODO: use keyword arguments on 0.7
    Bu = mean(transition_noise)      # = B_t * u_t   (input/control)
    Du = mean(observation_noise)     # = D_t * u_t
    Q = cov(transition_noise)
    R = cov(observation_noise)
    A = transition_mat
    C = observation_mat
    y = observation

    # Prediction step
    transitioned_state = A * state_prior + Bu
    μ = mean(transitioned_state)       # = μ_(t|t-1)
    Σ = cov(transitioned_state) + Q    # = Σ_(t|t-1)
    
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

function kalman_filter(initial_state_prior::Gaussian, observations::AbstractVector;
                       # "hidden" kwargs to help create defaults
                       _d=dim(initial_state_prior), _N=length(observations),
                       _d₂=length(observations[1]),
                       # Using `no_noise` twice makes the likelihood blow up.
                       # The Kalman filter needs at least _some_ noise.
                       transition_mats::AbstractVector=Fill(Eye(_d), _N),
                       transition_noises::AbstractVector{<:Gaussian}=Fill(no_noise(_d), _N),
                       observation_mats::AbstractVector=Fill(Eye(_d₂), _N),
                       observation_noises::AbstractVector{<:Gaussian}=Fill(no_noise(_d₂), _N))
    @assert(length(observations) == length(transition_mats) ==
            length(transition_noises) == length(observation_mats) ==
            length(observation_noises),
            "All passed vectors should be of the same length")
    state = initial_state_prior
    filtered_states = fill(initial_state_prior, _N)
    total_ll = 0.0
    for i in 1:length(observations)
        state, ll = kfilter(state, transition_mats[i], transition_noises[i],
                            observations[i], observation_mats[i], observation_noises[i])
        filtered_states[i] = state
        total_ll += ll
    end
    return (filtered_states, total_ll)
end

""" Compute the 1-step smoothed state, given the _next_ smoothed state. """
function ksmoother(filtered_state::Gaussian, next_filtered_state::Gaussian,
                   next_smoothed_state::Gaussian,
                   transition_mat::AbstractMatrix, transition_noise::Gaussian,
                   next_transition_mat::AbstractMatrix)
    # Deconstruct arguments
    Aₜ = transition_mat
    Aₜ₁ = next_transition_mat
    Bu = mean(transition_noise)      # = B_t * u_t   (input/control)
    Q = cov(transition_noise)
    μₜₜ, Σₜₜ = mean(filtered_state), cov(filtered_state)
    μₜ₁T, Σₜ₁T = mean(next_smoothed_state), cov(next_smoothed_state)

    # Predicted state
    transitioned_state = Aₜ₁ * filtered_state + Bu
    μₜ₁ₜ = mean(transitioned_state)       # = μ_(t|t-1)
    Σₜ₁ₜ = cov(transitioned_state) + Q    # = Σ_(t+1|t)

    # Smoothed state
    J = Σₜₜ * Aₜ₁ * Σₜ₁ₜ
    return Gaussian(μₜₜ + J * (μₜ₁T - μₜ₁ₜ),
                    Σₜₜ + J * (Σₜ₁T - Σₜ₁ₜ) * J')
end

function kalman_smoother(initial_state_prior::Gaussian, observations::AbstractVector;
                         # "hidden" kwargs to help create defaults
                         _d=dim(initial_state_prior), _N=length(observations),
                         _d₂=length(observations[1]),
                         # Using `no_noise` twice makes the likelihood blow up.
                         # The Kalman filter needs at least _some_ noise.
                         transition_mats::AbstractVector=Fill(Eye(_d), _N),
                         transition_noises::AbstractVector{<:Gaussian}=Fill(no_noise(_d), _N),
                         observation_mats::AbstractVector=Fill(Eye(_d₂), _N),
                         observation_noises::AbstractVector{<:Gaussian}=Fill(no_noise(_d₂), _N))
    filtered_states, ll = kalman_filter(initial_state_prior, observations;
                                        transition_mats=transition_mats,
                                        transition_noises=transition_noises,
                                        observation_mats=observation_mats,
                                        observation_noises=observation_noises)
    rev_smoothed_states = [filtered_states[end]]
    for t in length(observations)-1:-1:1
        push!(rev_smoothed_states,
              ksmoother(filtered_states[t], filtered_states[t+1],
                        rev_smoothed_states[end],
                        transition_mats[t], transition_noises[t], transition_mats[t+1]))
    end
    return filtered_states, reverse(rev_smoothed_states), ll
end

end  # module
