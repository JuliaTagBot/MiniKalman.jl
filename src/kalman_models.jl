# This code is 100% built _on top_ of MiniKalman.jl

using Unitful: ustrip, unit   # could be @required ?
using Optim
using QuickTypes: roottypeof  # TODO: get rid of dependency

export @kalman_model, sample_and_recover, optimize, marginal


################################################################################
marginal_var(g::Gaussian) = diag(cov(g))
marginal_var(g::Gaussian, i::Int) = diag(cov(g))[i]
marginal_std(args...) = sqrt(marginal_var(args...))
marginal(g::Gaussian, i::Int) = Gaussian(mean(g)[i], marginal_var(g, i))
is_marginal(g::Gaussian) = dim(g) == 1



################################################################################

full_initial_state(m) = make_full(initial_state(m))


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
# Optimization

split_units(vec::Vector) = ustrip.(vec), unit.(vec)

""" Finds a set of model parameters that attempts to maximize the log-likelihood
on the given dataset. Returns `(best_model, optim_object)`. """
function Optim.optimize(model0::Model, inputs,
                        observations::Union{Nothing, AbstractVector}=nothing;
                        initial_state=MiniKalman.initial_state(model0),
                        min=0.0, # 0.0 is a bit arbitrary...
                        parameters_to_optimize=fieldnames(typeof(model0)), 
                        method=LBFGS(linesearch=Optim.LineSearches.BackTracking()),
                        kwargs...)
    # It would be nice not to need split_units
    initial_x, units = split_units(get_params(model0, parameters_to_optimize))
    function objective(params)
        model = set_params(model0, params .* units, parameters_to_optimize)
        return -log_likelihood(model, inputs, observations; initial_state=initial_state)
    end
    td = OnceDifferentiable(objective, initial_x; autodiff=:forward)
    mins = min isa AbstractVector ? min : fill(min, length(initial_x))
    maxes = fill(Inf, length(initial_x))
    o = optimize(td, mins, maxes, initial_x, Fminbox(method), Optim.Options(; kwargs...))
    best_model = set_params(model0, Optim.minimizer(o) .* units, parameters_to_optimize)
    return (best_model, o)
end

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
    # show(io, MIME"text/html"(),
    #      plot_hidden_state(1:length(rr.obs), rr.estimated_state;
    #                        true_state=rr.true_state))
end

################################################################################
# Sampling

# This is essentially the definition of sampling from a dirac delta. 
Base.rand(RNG, P::Gaussian{U, Zero}) where U = P.μ

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

