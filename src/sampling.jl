# This is essentially the definition of sampling from a dirac delta. 
Base.rand(RNG, P::Gaussian{U, Zero}) where U = P.μ

function kalman_sample(m::Model, inputs, rng::AbstractRNG, start_state, N=length(inputs))
    # This code was optimized in 0.6, but sampling doesn't usually have to be
    # hyper-efficient, and it would look cleaner with a loop.
    result = accumulate(1:N; init=(start_state, nothing)) do v, t
        state, _ = v
        next_state = transition_mat(m, inputs, t) * state +
            rand(rng, transition_noise(m, inputs, t))
        return (next_state,
                observation_mat(m, inputs, t) * next_state +
                rand(rng, observation_noise(m, inputs, t)))
    end
    return (map(first, result), map(last, result))
end

################################################################################
# sample_and_recover

""" A data structure """
struct RecoveryResults
    true_model
    estimated_model
    true_state
    estimated_state
    obs
    optim
end
parameter_accuracy_ratios(rr::RecoveryResults) =
    [f=>getfield(rr.estimated_model, f) ./ getfield(rr.true_model, f)
     for f in fieldnames(typeof(rr.estimated_model))]

function Base.show(io::IO, ::MIME"text/html", rr::RecoveryResults)
    print(io, "Ratio of estimated/true parameters (1.0 is best): <br>")
    for (f, ratio) in parameter_accuracy_ratios(rr)
        print(io, "<pre>  ",
              f, " => ", round.(ratio, 4), 
              "</pre>")
    end
    # This functionality was neat, but requiring Plots is not nice.
    # show(io, MIME"text/html"(),
    #      plot_hidden_state(1:length(rr.obs), rr.estimated_state;
    #                        true_state=rr.true_state))
end

""" See if we can recover the model parameters _and_ the true parameters using
data generated from the model.

Concretely, we sample observations and hidden state from `true_model` for the
given `inputs`, then call `optimize` on `true_model * fuzz_factor`."""
function sample_and_recover(true_model::Model, inputs, rng;
                            parameters_to_optimize=fieldnames(typeof(true_model)),
                            fuzz_factor=exp.(randn(rng, length(get_params(true_model, parameters_to_optimize)))),
                            initial_state::Gaussian=initial_state(true_model),
                            start_model=nothing)
    rng = rng isa AbstractRNG ? rng : MersenneTwister(rng::Integer)
    true_state, obs = kalman_sample(true_model, inputs, rng, rand(rng, initial_state))
    if start_model === nothing
        start_model = set_params(true_model,
                                 (get_params(true_model, parameters_to_optimize) .*
                                  fuzz_factor),
                                 parameters_to_optimize)
    end
    (best_model, o) = optimize(start_model, inputs, obs;
                               initial_state=initial_state,
                               parameters_to_optimize=parameters_to_optimize)
    estimated_state = kalman_smoother(best_model, inputs, obs,
                                      initial_state=initial_state)
    return RecoveryResults(true_model, best_model, true_state, estimated_state, obs, o)
end
