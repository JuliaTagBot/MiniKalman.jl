# This is essentially the definition of sampling from a dirac delta. 
Base.rand(RNG::AbstractRNG, P::Gaussian{U, Zero}) where U = P.μ

function kalman_sample(m::Model, inputs, rng::AbstractRNG, start_state,
                       N::Integer=length(inputs))
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

""" A data structure containing the results of `sample_and_recover`. """
struct RecoveryResults
    true_model       # the true model, passed by the user
    estimated_model  # the model estimated from maximum likelihood
    true_samples     # the samples generated from the true model
    estimated_state  # the P(state|observation, estimated_model) distributions 
    obs              # the sampled observations
    optim            # the result from calling `optimize` (contains convergence details)
end
parameter_accuracy_ratios(rr::RecoveryResults) =
    [f=>getfield(rr.estimated_model, f) ./ getfield(rr.true_model, f)
     for f in parameters(rr.estimated_model)]

function Base.show(io::IO, rr::RecoveryResults)
    println(io, "Ratio of estimated/true parameters (1.0 is best): ")
    for (f, ratio) in parameter_accuracy_ratios(rr)
        println(io, "  ",
                f, " => ", round.(ratio*1000)/1000) 
    end
    # This functionality was neat, but requiring Plots is not nice.
    # show(io, MIME"text/html"(),
    #      plot_hidden_state(1:length(rr.obs), rr.estimated_state;
    #                        true_samples=rr.true_samples))
end

""" See if we can recover the model parameters _and_ the true parameters using
data generated from the model.

Concretely, we sample observations and hidden state from `true_model` for the
given `inputs`, then call `optimize` on `true_model * fuzz_factor`.

If `start_model` isn't specified, we start from a model in the neighborhood of 
`true_model` (with `fuzz_factor ~= 1.0` controlling how far we start).

We return a `RecoveryResults` object. See its definition for details. """
function sample_and_recover(true_model::Model, inputs, N=nothing; rng=GLOBAL_RNG,
                            parameters_to_optimize=parameters(true_model),
                            fuzz_factor=exp.(randn(rng, length(get_params(true_model, parameters_to_optimize)))),
                            initial_state::Gaussian=initial_state(true_model),
                            start_model=nothing,
                            input_f=(model, inp)->inp)
    rng = rng isa AbstractRNG ? rng : MersenneTwister(rng::Integer)
    inputs2 = get_inputs(true_model, inputs)
    N = N === nothing ? length(inputs2) : N
    true_samples, obs =
        kalman_sample(true_model, inputs2, rng, rand(rng, initial_state), N)
    log_likelihood(true_model, inputs2, obs)
    if start_model === nothing
        true_params = get_params(true_model, parameters_to_optimize)
        start_model = set_params(true_model, (true_params .* fuzz_factor),
                                 parameters_to_optimize)
    end
    (best_model, o) = optimize(start_model, inputs, obs;
                               initial_state=initial_state,
                               parameters_to_optimize=parameters_to_optimize)
    estimated_state = kalman_smoother(best_model, get_inputs(best_model, inputs), obs,
                                      initial_state=initial_state)
    return RecoveryResults(true_model, best_model, true_samples, estimated_state, obs, o)
end
