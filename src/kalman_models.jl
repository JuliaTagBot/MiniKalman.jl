# This code is 100% built _on top_ of MiniKalman.jl

using MacroTools
using MacroTools: postwalk
using Parameters
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

marginal_std(g::Gaussian) = sqrt.(diag(cov(g)))
marginal_std(g::Gaussian, i::Int) = sqrt(diag(cov(g))[i])


kalman_quantities = [:observation_mat, :observation_mats, 
                     :observation_noise, :observation_noises, 
                     :transition_mat, :transition_mats,
                     :transition_noise, :transition_noises,
                     :initial_state, :labels]
for q in kalman_quantities
    @eval function $q end  # forward declarations
end

@qstruct_fp EvaluatedInputs(observation_mats, observation_noises, transition_mats,
                            transition_noises, initial_state)
Base.length(ei::EvaluatedInputs) = length(ei.observation_mats)
const EInputs = Union{Inputs, EvaluatedInputs}
eval_inputs(::Model, ei::EvaluatedInputs) = ei
for q in fieldnames(EvaluatedInputs)
    @eval $q(::Model, ei::EvaluatedInputs) = $q(ei)
    @eval $q(ei::EvaluatedInputs) = ei.$q
end

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
export reconstruct   # Parameters.jl#57

""" Create a new model of the same type as `model`, but with the given `params`.
This is meant to be used with Optim.jl. Inspired from sklearn's `set_params`. """
function set_params(model::Model, params::AbstractVector, names=fieldnames(model))
    i = 1
    upd = Dict()
    for name in names
        v = getfield(model, name)
        nvals = length(v)
        upd[name] = params[v isa Number ? i : (i:i+nvals-1)]
        i += nvals
    end
    kwargs = map(fieldnames(model)) do f
        f=>get(upd, f) do
            getfield(model, f)
        end
    end
    # Assumes that there's a pure-kwarg constructor of the object
    return roottypeof(model)(; kwargs...)
end
get_params(model::Model, names=fieldnames(model)) =
    Float64[x for v in names for x in getfield(model, v)]


################################################################################
## Defaults
transition_mat(m, inputs, i) = Identity()
transition_mats(m, inputs) = mappedarray(i->transition_mat(m, inputs, i),
                                         1:length(inputs))
transition_noise(m, inputs, i) = no_noise()
transition_noises(m, inputs) = mappedarray(i->transition_noise(m, inputs, i),
                                           1:length(inputs))
# observation_noise(inputs::Inputs) = no_noise() is tempting, but it's
# a degenerate Kalman model, which causes problems
observation_noises(m, inputs) = mappedarray(i->observation_noise(m, inputs, i),
                                            1:length(inputs))
observation_mat(m, inputs, i) = Identity()
observation_mats(m, inputs) = mappedarray(i->observation_mat(m, inputs, i),
                                          1:length(inputs))


################################################################################
## Delegations

function kalman_filter(m::Model, inputs0::EInputs, observations::AbstractVector,
                       initial_state=MiniKalman.initial_state(m))
    inputs = eval_inputs(m, inputs0)
    kalman_filter(initial_state,
                  observations,
                  observation_noises(m, inputs),
                  transition_mats(m, inputs),
                  transition_noises(m, inputs),
                  observation_mats(m, inputs))
end
function log_likelihood(m::Model, inputs0::EInputs, observations::AbstractVector,
                        initial_state=MiniKalman.initial_state(m))
    inputs = eval_inputs(m, inputs0)
    log_likelihood(initial_state,
                   observations,
                   observation_noises(m, inputs),
                   transition_mats(m, inputs),
                   transition_noises(m, inputs),
                   observation_mats(m, inputs))
end

function kalman_smoother(m::Model, inputs0::EInputs,
                         filtered_states::AbstractVector{<:Gaussian})
    inputs = eval_inputs(m, inputs0)
    kalman_smoother(filtered_states;
                    transition_mats=transition_mats(m, inputs),
                    transition_noises=transition_noises(m, inputs))
end
function kalman_smoother(m::Model, inputs0::EInputs, observations::AbstractVector,
                         initial_state=MiniKalman.initial_state(m))
    inputs = eval_inputs(m, inputs0)
    kalman_smoother(m, inputs, kalman_filter(m, inputs, observations, initial_state)[1])
end

function kalman_sample(m::Model, inputs0::EInputs, rng::AbstractRNG,
                       initial_state=MiniKalman.initial_state(m))
    inputs = eval_inputs(m, inputs0)
    kalman_sample(rng, initial_state,
                  observation_noises(m, inputs);
                  transition_mats=transition_mats(m, inputs),
                  transition_noises=transition_noises(m, inputs),
                  observation_mats=observation_mats(m, inputs))
end

################################################################################
# Optimization

""" Finds a set of model parameters that attempts to maximize the log-likelihood
on the given dataset. Returns `(best_model, optim_object)`. """
function Optim.optimize(model0::Model, inputs::Inputs,
                        observations::AbstractVector,
                        initial_state=MiniKalman.initial_state(model0),;
                        min=0.0, # 0.0 is a bit arbitrary...
                        parameters_to_optimize=fieldnames(model0), 
                        method=LBFGS(linesearch=Optim.LineSearches.BackTracking()),
                        kwargs...)
    vars = parameters_to_optimize
    initial_x = get_params(model0, vars)
    function objective(params)
        model = set_params(model0, params, vars)
        return -log_likelihood(model, inputs, observations, initial_state)
    end
    td = OnceDifferentiable(objective, initial_x; autodiff=:forward)
    mins = min isa AbstractVector ? min : fill(min, length(initial_x))
    maxes = fill(Inf, length(initial_x))
    o = optimize(td, mins, maxes, initial_x, Fminbox(method), Optim.Options(; kwargs...))
    best_model = set_params(model0, Optim.minimizer(o), vars)
    return (best_model, o)
end


function plot_hidden_state!(p, time, estimates, i; true_state=nothing,
                            label="estimate", kwargs...)
    P = Main.Plots
    P.plot!(p, time, getindex.(mean.(estimates), i); label=label,
            ribbon=marginal_std.(estimates, i), msa=0.5, xlabel="time", kwargs...)
    if true_state !== nothing
        P.plot!(p, time, getindex.(true_state, i); label="truth",
                linestyle=:dash, color=:orange)
    end
    p
end
plot_hidden_state(a, b, c; kwargs...) =
    plot_hidden_state!(Main.Plots.plot(), a, b, c; kwargs...)
plot_hidden_state(time, estimates; true_state=nothing,
                  ylabels=["hidden_state[$i]" for i in 1:dim(estimates[1])],
                  kwargs...) =
    Main.Plots.plot([plot_hidden_state(time, estimates, i; true_state=true_state,
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
                            parameters_to_optimize=fieldnames(true_model),
                            fuzz_factor=exp.(randn(rng, length(get_params(true_model, parameters_to_optimize)))),
                            start_model=nothing)
    einputs = eval_inputs(true_model, inputs)
    state0 = initial_state(true_model)::Gaussian
    rng = rng isa AbstractRNG ? rng : MersenneTwister(rng::Integer)
    true_state, obs = kalman_sample(true_model, einputs, rng, rand(rng, state0))
    if start_model === nothing
        start_model = set_params(true_model,
                                 (get_params(true_model, parameters_to_optimize) .*
                                  fuzz_factor),
                                 parameters_to_optimize)
    end
    (best_model, o) = optimize(start_model, inputs, obs, state0;
                               parameters_to_optimize=parameters_to_optimize)
    estimated_state = kalman_smoother(best_model, inputs, obs, state0)
    return RecoveryResults(true_model, best_model, true_state, estimated_state, obs, o)
end

################################################################################

""" `Positive(Gaussian(...))` is a distribution that samples from `Gaussian`, but
rejects all negative samples. """
struct Positive   # JUN11: not used anymore
    distribution
end

