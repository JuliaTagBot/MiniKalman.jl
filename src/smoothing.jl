""" Compute the smoothed belief state at step `t`, given the `t+1`'th smoothed belief
state. Here, "next" === t+1 """
function ksmoother(filtered_state::Gaussian, next_smoothed_state::Gaussian,
                   next_transition_mat, next_transition_noise::Gaussian)
    # Again, taken from Murphy
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

function kalman_smoother!(out, m::Model, inputs, filtered_states;
                          steps=length(out)-1:-1:1)
    @assert steps[1] >= steps[end] "`steps` must be in descending order"
    for t in steps
        out[t] = ksmoother(filtered_states[t], out[t+1],
                           transition_mat(m, inputs, t+1),
                           transition_noise(m, inputs, t+1))
    end
    return out
end

""" Compute the smoothed estimates. Optionally, you can pass the filtered estimates
to save their computation cost. """
function kalman_smoother(m::Model, inputs, observations=nothing;
                         initial_state=initial_state(m),
                         filtered_states::AbstractVector=
                         kalman_filtered(m, inputs, observations;
                                         initial_state=initial_state)) 
    smoothed_states = fill(filtered_states[end], length(filtered_states))
    kalman_smoother!(smoothed_states, m, inputs, filtered_states)
    return smoothed_states
end
