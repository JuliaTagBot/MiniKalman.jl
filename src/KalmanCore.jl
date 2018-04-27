module KalmanCore

using GaussianDistributions, FillArrays
using GaussianDistributions: dim, logpdf
using StaticArrays

export kfilter, kalman_filter, white_noise

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

function kalman_filter(initial_state::Gaussian, observations::AbstractVector;
                       # "hidden" kwargs to help create defaults
                       _d=dim(initial_state), _N=length(observations),
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
    state = initial_state
    filtered_states = fill(initial_state, _N)
    total_ll = 0.0
    for (i, (observation, transition_mat, transition_noise, obs_mat, obs_noise)) in
        enumerate(zip(observations, transition_mats, transition_noises,
                      observation_mats, observation_noises))
        state, ll = kfilter(state, transition_mat, transition_noise,
                            observation, obs_mat, obs_noise)
        filtered_states[i] = state
        total_ll += ll
    end
    return (filtered_states, total_ll)
end



end  # module
