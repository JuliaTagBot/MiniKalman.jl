using MacroTools
using Parameters
using Optim
import GaussianDistributions

include("identities.jl")

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


kalman_quantities = [:observation_mat, :observation_mats, #:initial_state,
                     :observation_noise, :observation_noises, 
                     :transition_mat, :transition_mats,
                     :transition_noise, :transition_noises,
                     :labels]
for q in kalman_quantities
    @eval function $q end  # forward declarations
end


""" See notebook 06 for examples. """
macro kalman_model(def)
    @assert(@capture(def, model_type_(; params__) do input_vars__; qtydefs__ end),
            "Use @kalman_model M(; param1=..., param2=...) do input1, input2, ... end)")
    inputs_type = Symbol(model_type, "Inputs")
    param_vars = map(first ∘ splitarg, params)

    @gensym km ki
    fundefs = map(qtydefs) do c
        @assert(@capture(c, fname_ := expr_), "Bad quantity definition: $c")
        @assert(fname in kalman_quantities,
                "`$fname` is not a valid Kalman model quantity ($kalman_quantities)")
        if fname == :labels
            quote
                $MiniKalman.labels($km::$model_type) = $expr
            end
        else
            quote
                $MiniKalman.$fname($km::$model_type, $ki::$MiniKalman.Inputs) =
                    # We break $fname in two definitions, because `expr` should
                    # be evaluated in an environment where all types are known.
                    $MiniKalman.$fname($km, nothing,
                                       $([:($km.$p) for p in param_vars]...),
                                       $([:($ki[$(Expr(:quote, i))])
                                          for i in input_vars]...))
                $MiniKalman.$fname($km::$model_type, ::Void,
                                   $(param_vars...), $(input_vars...))=
                $expr  # the computation is done here
            end
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
        $(fundefs...)
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
    return typeof(model)(model; upd...)
end
get_params(model::Model, names=fieldnames(model)) =
    Float64[x for v in names for x in getfield(model, v)]


################################################################################
## Defaults
transition_mat(m, inputs::Inputs) = Identity()
transition_mats(m, inputs::Inputs) = Fill(transition_mat(m, inputs), length(inputs))
transition_noise(m, inputs::Inputs) = no_noise()
transition_noises(m, inputs::Inputs) = Fill(transition_noise(m, inputs), length(inputs))
# observation_noise(inputs::Inputs) = no_noise() is tempting, but it's
# a degenerate Kalman model, which causes problems
observation_noises(m, inputs::Inputs) = Fill(observation_noise(m, inputs), length(inputs))
observation_mat(m, inputs::Inputs) = Identity()
observation_mats(m, inputs::Inputs) = Fill(observation_mat(m, inputs), length(inputs))


################################################################################
## Delegations

kalman_filter(m::Model, inputs::Inputs, observations::AbstractVector, initial_state) = 
    kalman_filter(initial_state, observations,
                  observation_noises(m, inputs),
                  transition_mats(m, inputs),
                  transition_noises(m, inputs),
                  observation_mats(m, inputs))

kalman_smoother(m::Model, inputs::Inputs, filtered_states::AbstractVector{<:Gaussian}) =
    kalman_smoother(filtered_states;
                    transition_mats=transition_mats(m, inputs),
                    transition_noises=transition_noises(m, inputs))
kalman_smoother(m::Model, inputs::Inputs, observations::AbstractVector,
                initial_state::Gaussian) =
    kalman_smoother(m, inputs, kalman_filter(m, inputs, observations, initial_state)[1])

kalman_sample(m::Model, inputs::Inputs, rng::AbstractRNG, initial_state) =
    kalman_sample(rng, initial_state, observation_noises(m, inputs);
                  transition_mats=transition_mats(m, inputs),
                  transition_noises=transition_noises(m, inputs),
                  observation_mats=observation_mats(m, inputs))

################################################################################
# Optimization

""" Finds a set of model parameters that attempts to maximize the log-likelihood
on the given dataset. Returns `(best_model, optim_object)`. """
function Optim.optimize(model0::Model, inputs::Inputs,
                        observations::AbstractVector, initial_state;
                        min=0.0, # 0.0 is a bit arbitrary...
                        parameters_to_optimize=fieldnames(model0), post_f=identity,
                        kwargs...)
    @assert isempty(kwargs) # should be passed to Optim.Options
    vars = parameters_to_optimize
    initial_x = get_params(model0, vars)
    function objective(params)
        model = post_f(set_params(model0, params, vars))
        _, ll = kalman_filter(model, inputs, observations, initial_state)
        return -ll
    end
    td = OnceDifferentiable(objective, initial_x; autodiff=:forward)
    mins = min isa AbstractVector ? min : fill(min, length(initial_x))
    maxes = fill(Inf, length(initial_x))
    o = optimize(td, initial_x, mins, maxes, Fminbox{LBFGS}())
    best_model = set_params(model0, Optim.minimizer(o), vars)
    return (best_model, o)
end


function plot_hidden_state!(p, time, estimates, i; true_state=nothing,
                            label="estimate", kwargs...)
    P.plot!(p, time, getindex.(mean.(estimates), i), label=label, 
            ribbon=marginal_std.(estimates, i), msa=0.5, xlabel="time", kwargs...)
    if true_state !== nothing
        P.plot!(p, time, getindex.(true_state, i); label="truth",
                linestyle=:dash, color=:orange)
    end
    p
end
plot_hidden_state(p, args...; kwargs...) =
    plot_hidden_state!(Main.Plots.plot(), args...; kwargs...)
function plot_hidden_state(time, estimates; true_state=nothing,
                           ylabels=["hidden_state[$i]" for i in 1:dim(estimates[1])],
                           kwargs...)
    P = Main.Plots
    P.plot([plot_hidden_state(time, estimates, i; true_state=true_state,
                              ylabel=ylabels[i], kwargs...)
            for i in 1:dim(estimates[1])]...)
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
    show(io, MIME"text/html"(),
         plot_hidden_state(1:length(rr.obs), rr.estimated_state;
                           true_state=rr.true_state))
end

""" See if we can recover the model parameters _and_ the true parameters using
data generated from the model.

Concretely, we sample observations and hidden state from `true_model` for the
given `inputs`, then call `optimize` on `true_model * fuzz_factor`."""
function sample_and_recover(true_model::Model, inputs::Inputs,
                            rng, initial_state::Gaussian;
                            fuzz_factor=exp.(randn(rng, length(get_params(true_model)))),
                            start_model=nothing)
    rng = rng isa AbstractRNG ? rng : MersenneTwister(rng::Integer)
    true_state, obs = kalman_sample(true_model, inputs, rng, rand(rng, initial_state))
    if start_model === nothing
        start_model = set_params(true_model, get_params(true_model) .* fuzz_factor)
    end
    (best_model, o) = optimize(start_model, inputs, obs, initial_state)
    estimated_state = kalman_smoother(best_model, inputs, obs, initial_state)
    return RecoveryResults(true_model, best_model, true_state, estimated_state, obs, o)
end

################################################################################

""" `Positive(Gaussian(...))` is a distribution that samples from `Gaussian`, but
rejects all negative samples. """
struct Positive
    distribution
end

