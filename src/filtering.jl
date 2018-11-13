kalman_quantities = [:observation_mat, :observation_noise,
                     :transition_mat, :transition_noise,
                     :initial_state, :observation, :labels]
for q in kalman_quantities
    @eval function $q end  # forward declarations
end

################################################################################
# Model

""" `Model` objects assume that there is a pure-kwarg constructors (such as provided by
Parameters.jl) """
abstract type Model end

# Defaults
transition_mat(m, inputs, i) = Identity()
transition_noise(m, inputs, i) = Zero()
observation_mat(m, inputs, i) = Identity()

################################################################################

predicted_state(state_prior::Gaussian, transition_mat, transition_noise::Gaussian) =
    # Helper. Returning a tuple is more convenient than a Gaussian
    (transition_mat * mean(state_prior) + mean(transition_noise),
     transition_mat * cov(state_prior) * transition_mat' + cov(transition_noise))

""" Perform one step of Kalman filtering, for online use. We assume equations:

```julia
current_state = transition_mat * state_prior + transition_noise
observation = observation_mat * current_state + observation_noise
```

and return this tuple:

```
   (current_state::Gaussian,   # the posterior `P(state|observation)`
    ll,                        # the log-likelihood for that step
    predicted_obs::Gaussian)   # `P(observation|state)`
```

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

kfilter(prev_state::Gaussian, m::MiniKalman.Model, inp, t::Int, observations=nothing) = 
    kfilter(prev_state, transition_mat(m, inp, t),
            transition_noise(m, inp, t),
            observations===nothing ? observation(inp, t) : observations[t],
            observation_mat(m, inp, t), observation_noise(m, inp, t))

""" (Internal function) return three vectors, appropriate for storing 
states, likelihoods, and P(observation). """
function output_vectors(m::Model, einputs, observations=nothing; length=length(einputs),
                        initial_state=initial_state(m))
    state = make_full(initial_state)
    # For type stability, we fake-run it. It's rather lame. Ideally, we'd build all
    # output types from the input types
    state2, _, dum_predictive = kfilter(state, m, einputs, 1, observations)
    @assert typeof(state) == typeof(state2)
    filtered_states = Vector{typeof(state)}(undef, length)
    predicted_obs = Vector{typeof(dum_predictive)}(undef, length)
    lls = Vector{Float64}(undef, length)
    return (filtered_states, lls, predicted_obs)
end

function kalman_filter!(filtered_states::AbstractVector, lls::AbstractVector,
                        predicted_obs::AbstractVector,
                        m::Model, inputs, observations=nothing;
                        steps::AbstractRange=1:length(filtered_states),
                        initial_state=(steps[1]==1 ? initial_state(m) :
                                       filtered_states[steps[1]-1]))
    state = make_full(initial_state)  # we need make_full so that the state does
                                      # not change type during iteration
    for t in steps
        state, lls[t], predicted_obs[t] = kfilter(state, m, inputs, t, observations)
        filtered_states[t] = state
    end
end

function kalman_filter(m::Model, inputs, observations=nothing;
                       initial_state=initial_state(m), steps=1:length(inputs))
    out_vecs = output_vectors(m, inputs, observations, length=length(steps))
    kalman_filter!(out_vecs..., m, inputs, observations; steps=steps,
                   initial_state=initial_state)
    return out_vecs
end

function log_likelihood(m::Model, inputs, observations=nothing;
                        initial_state=MiniKalman.initial_state(m), steps=1:length(inputs))
    # Since this is in the inner loop of `optimize`, we make sure it's non-allocating
    ll_sum = 0.0
    state = make_full(initial_state)
    for t in steps
        state, ll, _ = kfilter(state, m, inputs, t, observations)
        ll_sum += ll
    end
    return ll_sum
end


kalman_filtered(args...; kwargs...) = kalman_filter(args...; kwargs...)[1]  # convenience

""" Convenience; returns a vector of the total likelihood up to each step. """
cumulative_log_likelihood(args...; kwargs...) =
    cumsum(kalman_filter(args...; kwargs...)[2])
