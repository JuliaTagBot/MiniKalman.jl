using MacroTools, QuickTypes
using QuickTypes: fieldsof, construct
using Optim

include("identities.jl")

export @kalman_model, Positive

abstract type Model end
abstract type Inputs end

struct Unspecified end


kalman_quantities = [:observation_mat, :observation_mats, #:initial_state,
                     :observation_noise, :observation_noises, 
                     :transition_mat, :transition_mats,
                     :transition_noise, :transition_noises]
for q in kalman_quantities
    @eval function $q end
end
function input_type end


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
        quote
            function $MiniKalman.$fname($km::$model_type, $ki::$inputs_type)
                $([:($p = $km.$p) for p in param_vars]...)
                $([:($i = $ki.$i) for i in input_vars]...)
                $expr
            end
        end
    end

    unspecified_kw(k) = Expr(:kw, k, :($MiniKalman.Unspecified()))

    esc(quote
        $MiniKalman.@qstruct $model_type(; $(map(unspecified_kw, param_vars)...)) <: $MiniKalman.Model
        $MiniKalman.@qstruct $inputs_type(_N::Int;
                                          $(map(unspecified_kw, input_vars)...)) <: $MiniKalman.Inputs
        $(fundefs...)
        $MiniKalman.input_type(::Type{<:$model_type}) = $inputs_type
        $model_type
    end)
end

Base.length(inputs::Inputs) = inputs._N
""" Create a new model of the same type as `model`, but with the given `params`.
This is meant to be used with Optim.jl. Inspired from sklearn's `set_params`. """
function set_params(model::Model, params::AbstractVector)
    i = [0]
    next() = params[i[1]+=1]
    return construct(typeof(model),
                     [v isa AbstractVector ? map(_->next(), v) : next()
                      for v in fieldsof(model)]...)
end
get_params(model::Model) = Float64[x for v in fieldsof(model) for x in v]


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

# Rename to `Inputs(model; kwargs...)`?
Inputs(model::Model, N::Int; kwargs...) = input_type(typeof(model))(N; kwargs...)

kalman_filter(m::Model, inputs::Inputs, observations::AbstractVector, initial_state) = 
    kalman_filter(initial_state, observations,
                  observation_noises(m, inputs);
                  transition_mats=transition_mats(m, inputs),
                  transition_noises=transition_noises(m, inputs),
                  observation_mats=observation_mats(m, inputs))

kalman_smoother(m::Model, inputs::Inputs, filtered_states::AbstractVector{<:Gaussian})=
    kalman_smoother(filtered_states;
                    transition_mats=transition_mats(m, inputs),
                    transition_noises=transition_noises(m, inputs))
kalman_smoother(m::Model, inputs::Inputs, observations::AbstractVector,
                initial_state::Gaussian) =
    kalman_smoother(inputs, kalman_filter(m, inputs, observations, initial_state)[1])

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
                        observations::AbstractVector, initial_state)
    initial_x = get_params(model0)
    function objective(params)
        model = set_params(model0, params)
        -kalman_filter(model, inputs, observations, initial_state)[2]
    end
    td = OnceDifferentiable(objective, initial_x; autodiff=:forward)
    mins = fill(0.0, length(initial_x))
    maxes = fill(Inf, length(initial_x))
    o = optimize(td, initial_x, mins, maxes, Fminbox{LBFGS}())
    best_model = set_params(model0, Optim.minimizer(o))
    return (best_model, o)
end

################################################################################

""" `Positive(Gaussian(...))` is a distribution that samples from `Gaussian`, but
rejects all negative samples. """
struct Positive
    distribution
end

