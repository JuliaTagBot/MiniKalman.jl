module KalmanCore

using GaussianDistributions, FillArrays

""" Perform one step of Kalman filtering. We assume equations:

```julia
current_state = transition_mat * previous_state + transition_noise
observation = observation_mat * current_state + observation_noise
```

and return `current_state::Gaussian`, which is the posterior `P(state|observation)`.

To add a forcing term, pass it as `transition_noise = Gaussian(forcing_term, noise_cov)`.
"""
function kfilter(previous_state::Gaussian, observation::AbstractVector,
                 transition_mat::AbstractMatrix, transition_noise::Gaussian,
                 observation_mat::AbstractMatrix, observation_noise::Gaussian)
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

    # Prediction step. 
    predicted_state = M * previous_state + Bu
    μ = mean(predicted_state)       # = μ_(t|t-1)
    Σ = cov(predicted_state) + Q    # = Σ_(t|t-1)
    
    # Filter
    S = C * Σ * C' + R
    K = Σ * C' / S         # Kalman gain matrix
    ŷ = C * μ + Du
    r = y - ŷ
    ll = logpdf(Gaussian(C * μ, S), y)  # log-likelihood
    return (Gaussian(μ + K*r,
                     (I - K*C) * Σ),
            ll)
end

function kalman_filter(initial_state::Gaussian, observations::AbstractVector;
                       # "hidden" variables to create defaults
                       _d=size(initial_state), _N=length(observations),
                       _d₂=length(observations[1]),
                       transition_mats::AbstractVector=Fill(Eye(_d), _N),
                       transition_noises::AbstractVector{Gaussian}=Fill(no_noise(_d), _N),
                       observation_mats::AbstractVector=Fill(Gaussian, _N),
                       observation_noises::AbstractVector{Gaussian}=Fill(no_noise(_d₂), _N))
    @assert(length(observations) == length(transition_mats) ==
            length(transition_noises) == length(observation_mats) ==
            length(observation_noises),
            "All passed vectors should be of the same length")
    state = initial_state
    filtered_states = fill(initial_state, N)
    total_ll = 0.0
    for (i, (observation, transition_mat, transition_noise, obs_mat, obs_noise)) in
        enumerate(zip(observations, transition_mats, transition_noises,
                      observation_mats, observation_noises))
        filtered_states[i], ll = kalman_filter(state, observation,
                                               transition_mat, transition_noise,
                                               obs_mat, obs_noise)
        total_ll += ll
    end
    return (filtered_states, ll)
end

end  # module
