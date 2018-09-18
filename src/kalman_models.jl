# This code is 100% built _on top_ of MiniKalman.jl

using MacroTools
using MacroTools: postwalk
using Parameters  # for @with_kw
using Optim
using QuickTypes
using QuickTypes: roottypeof
import GaussianDistributions

export @kalman_model, sample_and_recover, optimize

abstract type Model end

################################################################################
# Inputs

abstract type Inputs end

# We have to store N, because some Kalman filters simply have no inputs, or the inputs
# are not 
""" A simple structure to store the length of the time series (`N`) + whatever
inputs the model calls for. This does not contain the observations. """
struct DictInputs <: Inputs
    N::Int
    quantities::Dict{Symbol, Any}
end
Inputs(N::Int; kwargs...) = DictInputs(N, Dict(kwargs))
Base.getindex(inputs::DictInputs, sym::Symbol) = 
    (haskey(inputs.quantities, sym) ? inputs.quantities[sym] :
     error("Missing input $sym"))
Base.length(inputs::DictInputs) = inputs.N

################################################################################

marginal_var(g::Gaussian) = diag(cov(g))
marginal_var(g::Gaussian, i::Int) = diag(cov(g))[i]
marginal_std(args...) = sqrt(marginal_var(args...))
marginal(g::Gaussian, i::Int) = Gaussian(mean(g)[i], marginal_var(g, i))
is_marginal(g::Gaussian) = dim(g) == 1

kalman_quantities = [:observation_mat, :observation_mats, 
                     :observation_noise, :observation_noises, 
                     :transition_mat, :transition_mats,
                     :transition_noise, :transition_noises,
                     :initial_state, :observation, :labels]
for q in kalman_quantities
    @eval function $q end  # forward declarations
end

# @qstruct_fp EvaluatedInputs(observation_mats, observation_noises, transition_mats,
#                             transition_noises, initial_state)
# Base.length(ei::EvaluatedInputs) = length(ei.observation_mats)
# const Inputs = Union{Inputs, EvaluatedInputs}
# eval_inputs(::Model, ei::EvaluatedInputs) = ei
# for q in fieldnames(EvaluatedInputs)
#     @eval $q(::Model, ei::EvaluatedInputs) = $q(ei)
#     @eval $q(ei::EvaluatedInputs) = ei.$q
# end

singular_qty_defaults = quote
    transition_mat = $MiniKalman.Identity()
    transition_noise = $MiniKalman.no_noise()
    observation_mat = $MiniKalman.Identity()
end

# singular_qty_defaults =
#     Dict(:transition_mat=>:($MiniKalman.Identity()),
#          :transition_noise=>:($MiniKalman.no_noise()),
#          :observation_mat=>:($MiniKalman.Identity()))

qty_defaults = Dict(:transition_mats=>:($MiniKalman.Fill(transition_mat, _N)),
                    :transition_noises=>:($MiniKalman.Fill(transition_noise, _N)),
                    :observation_noises=>:($MiniKalman.Fill(observation_noise, _N)),
                    :observation_mats=>:($MiniKalman.Fill(observation_mat, _N)))

""" See notebook 06 for examples. """
macro kalman_model(def)
    @assert(@capture(def, model_type_(; params__) do input_vars__; qtydefs0_ end),
            "Use @kalman_model M(; param1=..., param2=...) do input1, input2, ... end)")
    inputs_type = Symbol(model_type, "Inputs")
    param_vars = map(first ∘ splitarg, params)

    @gensym km ki
    defined = []  # TODO: use @defined in 0.7
    label_def = nothing
    qtydefs = postwalk(qtydefs0) do x
        # Turn := into =. := is essentially deprecated (July'18)
        if @capture(x, a_ := b_)
            push!(defined, a)
            if a == :labels
                label_def = 
                    quote
                        $MiniKalman.labels($km::$model_type) = $b
                    end
            end
            :($a = $b)
        else
            x
        end
    end

    proc_param(s::Symbol) = s
    proc_param(e::Expr) =
        # Replace :kw with :=, because that's what @with_kw expects
        (e.head==:kw) ? Expr(:(=), e.args...) : e
    esc(quote
        $MiniKalman.@with_kw struct $model_type <: $(MiniKalman.Model) # Parameters.jl#56
            $(map(proc_param, params)...)
        end
        $label_def
        $MiniKalman.eval_inputs($km::$model_type, $ki::$MiniKalman.Inputs) =
            # We break $eval_inputs in two definitions, because `expr` should
            # be evaluated in an environment where all types are known.
            $MiniKalman.eval_inputs($km, nothing, $ki,
                                    $([:($km.$p) for p in param_vars]...),
                                    $([:($ki[$(Expr(:quote, i))])
                                       for i in input_vars]...))
        function $MiniKalman.eval_inputs($km::$model_type, ::Void,
                                         $ki::$MiniKalman.Inputs,
                                         $(param_vars...), $(input_vars...))
            _N = length($ki)
            $singular_qty_defaults
            # $([:($qty = nothing) for qty in fieldnames(EvaluatedInputs)]...)
            $qtydefs
            # $MiniKalman.EvaluatedInputs($(fieldnames(EvaluatedInputs)...))
            $MiniKalman.EvaluatedInputs($([qty in defined ? qty : qty_defaults[qty]
                                           for qty in fieldnames(EvaluatedInputs)]...))
        end
        $model_type
    end)
end

""" Create a new model of the same type as `model`, but with the given `params`.
This is meant to be used with Optim.jl. Inspired from sklearn's `set_params`. """
function set_params(model::Model, params::AbstractVector, names=fieldnames(typeof(model)))
    i = 1
    upd = Dict()
    for name in names
        v = getfield(model, name)
        nvals = length(v)
        upd[name] = params[v isa Number ? i : (i:i+nvals-1)]
        i += nvals
    end
    kwargs = map(fieldnames(typeof(model))) do f
        f=>get(upd, f) do
            getfield(model, f)
        end
    end
    # Assumes that there's a pure-kwarg constructor of the object
    return roottypeof(model)(; kwargs...)
end
get_params(model::Model, names=fieldnames(typeof(model))) =
    Float64[x for v in names for x in getfield(model, v)]


################################################################################
## Defaults
transition_mat(m, inputs, i) = Identity() #transition_noises(m, inputs)[i]
#transition_mats(m, inputs) = Fill(Identity(), length(inputs))
transition_noise(m, inputs, i) = Zero() #transition_noises(m, inputs)[i]
#transition_noises(m, inputs) = Fill(Zero(), length(inputs))
# observation_noise(inputs::Inputs) = no_noise() is tempting, but it's
# a degenerate Kalman model, which causes problems
#observation_noise(m, inputs, i) = observation_noises(m, inputs)[i]
observation_mat(m, inputs, i) = Identity() #observation_mats(m, inputs)[i]
#observation_mats(m, inputs) = Fill(Identity(), length(inputs))


################################################################################
## Delegations

full_initial_state(m) = make_full(initial_state(m))

kfilter(prev_state::Gaussian, m::MiniKalman.Model, inp, t::Int, observations=nothing) = 
    kfilter(prev_state, transition_mat(m, inp, t),
            transition_noise(m, inp, t),
            observations===nothing ? observation(inp, t) : observations[t],
            observation_mat(m, inp, t), observation_noise(m, inp, t))

function kalman_filter!(filtered_states::AbstractVector, lls::AbstractVector,
                        predicted_obs::AbstractVector,
                        m::Model, inputs::Inputs, observations=nothing,
                        steps::AbstractRange=1:length(filtered_states),
                        initial_state=(steps[1]==1 ? full_initial_state(m) :
                                       filtered_states[steps[1]-1]))
    state = make_full(initial_state)  # we need make_full so that the state does
                                      # not change type during iteration
    for t in steps
        state, lls[t], predicted_obs[t] = kfilter(state, m, inputs, t, observations)
        filtered_states[t] = state
    end
end

function output_vectors(m::Model, einputs, observations=nothing)
    state = full_initial_state(m)
    # For type stability, we fake-run it. It's rather lame. Ideally, we'd build all
    # output types from the input types
    state2, _, dum_predictive = kfilter(state, m, einputs, 1, observations)
    @assert typeof(state) == typeof(state2)
    filtered_states = Vector{typeof(state)}(undef, length(einputs))
    predicted_obs = Vector{typeof(dum_predictive)}(undef, length(einputs))
    lls = Vector{Float64}(undef, length(einputs))
    return (filtered_states, lls, predicted_obs)
end

function kalman_filter(m::Model, inputs::Inputs, observations=nothing,
                       initial_state=initial_state(m))
    out_vecs = output_vectors(m, inputs, observations)
    kalman_filter!(out_vecs..., m, inputs, observations, 1:length(out_vecs[1]),
                   initial_state)
    return out_vecs
end

function log_likelihood(m::Model, inputs::Inputs, observations=nothing,
                        initial_state=MiniKalman.initial_state(m))
    ll_sum = 0.0
    state = make_full(initial_state)
    for t in 1:length(inputs)
        state, ll, _ = kfilter(state, m, inputs, t, observations)
        ll_sum += ll
    end
    return ll_sum
end

map_i(f, m, inputs) = mappedarray(i->f(m, inputs, i), 1:length(inputs))

function kalman_smoother!(smoothed_states, m::Model, inputs, filtered_states;
                          steps=length(smoothed_states)-1:-1:1)
    @assert steps[1] >= steps[end]
    for t in steps
        smoothed_states[t] =
              ksmoother(filtered_states[t], smoothed_states[t+1],
                        transition_mat(m, inputs, t+1),
                        transition_noise(m, inputs, t+1))
    end
end

function kalman_smoother(m::Model, inputs::Inputs,
                         filtered_states::AbstractVector{<:Gaussian})
    smoothed_states = fill(filtered_states[end], length(filtered_states))
    kalman_smoother!(smoothed_states, m, inputs, filtered_states)
    return smoothed_states
end
kalman_smoother(m::Model, inputs::Inputs, observations=nothing,
                initial_state=MiniKalman.initial_state(m)) =
    kalman_smoother(m, inputs, kalman_filtered(m, inputs, observations, initial_state))

function kalman_sample(m::Model, inputs::Inputs, rng::AbstractRNG,
                       initial_state=MiniKalman.initial_state(m))
    kalman_sample(rng, initial_state,
                  map_i(observation_noise, m, inputs);
                  transition_mats=map_i(transition_mat, m, inputs),
                  transition_noises=map_i(transition_noise, m, inputs),
                  observation_mats=map_i(observation_mat, m, inputs))
end

################################################################################
# Optimization

""" Finds a set of model parameters that attempts to maximize the log-likelihood
on the given dataset. Returns `(best_model, optim_object)`. """
function Optim.optimize(model0::Model, inputs::Inputs,
                        observations::Union{Nothing, AbstractVector}=nothing;
                        initial_state=MiniKalman.initial_state(model0),
                        min=0.0, # 0.0 is a bit arbitrary...
                        parameters_to_optimize=fieldnames(typeof(model0)), 
                        method=LBFGS(linesearch=Optim.LineSearches.BackTracking()),
                        kwargs...)
    initial_x = get_params(model0, parameters_to_optimize)
    function objective(params)
        model = set_params(model0, params, parameters_to_optimize)
        return -log_likelihood(model, inputs, observations, initial_state)
    end
    td = OnceDifferentiable(objective, initial_x; autodiff=:forward)
    mins = min isa AbstractVector ? min : fill(min, length(initial_x))
    maxes = fill(Inf, length(initial_x))
    o = optimize(td, mins, maxes, initial_x, Fminbox(method), Optim.Options(; kwargs...))
    best_model = set_params(model0, Optim.minimizer(o), parameters_to_optimize)
    return (best_model, o)
end

function plot_hidden_state!(p, time, marginals; true_state=nothing,
                            label="estimate", kwargs...)
    P = Main.Plots
    @assert is_marginal(marginals[1]) "Must pass a marginal gaussian."
    P.plot!(p, time, first.(mean.(marginals)); label=label,
            ribbon=first.(sqrt.(cov.(marginals))), msa=0.5, xlabel="time", kwargs...)
    if true_state !== nothing
        P.plot!(p, time, getindex.(true_state, i); label="truth",
                linestyle=:dash, color=:orange)
    end
    p
end
# plot_hidden_state(a, b; kwargs...) =
#     plot_hidden_state!(Main.Plots.plot(), a, b; kwargs...)
plot_hidden_state(time, estimates; true_state=nothing,
                  ylabels=["hidden_state[$i]" for i in 1:dim(estimates[1])],
                  kwargs...) =
    Main.Plots.plot([plot_hidden_state(time, marginal(estimates, i);
                                       true_state=true_state,
                                       ylabel=ylabels[i], kwargs...)
                     for i in 1:dim(estimates[1])]...)



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
    show(io, MIME"text/html"(),
         plot_hidden_state(1:length(rr.obs), rr.estimated_state;
                           true_state=rr.true_state))
end

""" See if we can recover the model parameters _and_ the true parameters using
data generated from the model.

Concretely, we sample observations and hidden state from `true_model` for the
given `inputs`, then call `optimize` on `true_model * fuzz_factor`."""
function sample_and_recover(true_model::Model, inputs::Inputs, rng;
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
    estimated_state = kalman_smoother(best_model, inputs, obs, initial_state)
    return RecoveryResults(true_model, best_model, true_state, estimated_state, obs, o)
end

################################################################################

""" `Positive(Gaussian(...))` is a distribution that samples from `Gaussian`, but
rejects all negative samples. """
struct Positive   # JUN11: not used anymore
    distribution
end

