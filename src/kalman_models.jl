using MacroTools, QuickTypes

include("identities.jl")

export @kalman_model, Positive

abstract type KalmanModel end
abstract type KalmanInputs end

struct Unspecified end


kalman_quantities = [:observations,
                     :observation_mat, :observation_mats, #:initial_state,
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

    @gensym km
    fundefs = map(qtydefs) do c
        @assert(@capture(c, fname_ := expr_), "Bad quantity definition: $c")
        @assert(fname in kalman_quantities,
                "`$fname` is not a valid Kalman model quantity ($kalman_quantities)")
        quote
            function $MiniKalman.$fname($km::$inputs_type)
                $([:($p = $km.model.$p) for p in param_vars]...)
                $([:($i = $km.$i) for i in input_vars]...)
                $expr
            end
        end
    end

    unspecified_kw(k) = Expr(:kw, k, :($MiniKalman.Unspecified()))

    esc(quote
        $MiniKalman.@qstruct $model_type(; $(map(unspecified_kw, param_vars)...)) <: $MiniKalman.KalmanModel
        $MiniKalman.@qstruct $inputs_type(model::$model_type;
                                          $(map(unspecified_kw, input_vars)...)) <: $MiniKalman.KalmanInputs
        $(fundefs...)
        $MiniKalman.input_type(::Type{<:$model_type}) = $inputs_type
        $model_type
    end)
end

Base.length(inputs::KalmanInputs) = length(observations(inputs))

## Defaults
transition_mat(inputs::KalmanInputs) = IdentityMat()
transition_mats(inputs::KalmanInputs) = Fill(transition_mat(inputs), length(inputs))
transition_noise(inputs::KalmanInputs) = no_noise()
transition_noises(inputs::KalmanInputs) = Fill(transition_noise(inputs), length(inputs))
# observation_noise(inputs::KalmanInputs) = no_noise() is tempting, but it's
# a degenerate Kalman model, which causes problems
observation_noises(inputs::KalmanInputs) = Fill(observation_noise(inputs), length(inputs))
observation_mat(inputs::KalmanInputs) = IdentityMat()
observation_mats(inputs::KalmanInputs) = Fill(observation_mat(inputs), length(inputs))

## Delegations

function kalman_filter(model::KalmanModel, initial_state::Gaussian; kwargs...)
    inputs = input_type(typeof(model))(model; kwargs...)
    kalman_filter(initial_state, observations(inputs),
                  observation_noises(inputs);
                  transition_mats=transition_mats(inputs),
                  transition_noises=transition_noises(inputs),
                  observation_mats=observation_mats(inputs))
end


################################################################################

""" `Positive(Gaussian(...))` is a distribution that samples from `Gaussian`, but
rejects all negative samples. """
struct Positive
    distribution
end

