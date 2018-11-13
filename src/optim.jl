using Optim

""" Create a new model of the same type as `model`, but with the given `params`.
This is meant to be used with Optim.jl. Inspired from sklearn's `set_params`. """
function set_params(model::Model, params::AbstractVector, names=fieldnames(typeof(model)))
    # Not efficient, but doesn't really have to be for significant input length.
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
    return roottypeof(model)(; kwargs...)
end
get_params(model::Model, names=fieldnames(typeof(model))) =
    [x for v in names for x in getfield(model, v)]

################################################################################

split_units(vec::Vector) = ustrip.(vec), unit.(vec)

""" Finds a set of model parameters that attempts to maximize the log-likelihood
on the given dataset. Returns `(best_model, optim_output_object)`. """
function Optim.optimize(model0::Model, inputs,
                        observations::Union{Nothing, AbstractVector}=nothing;
                        initial_state=MiniKalman.initial_state(model0),
                        min=0.0, # 0.0 is arbitrary... see below
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
    # Tell Optim that no parameter can be below `mins`. This should be made optional, TODO
    mins = min isa AbstractVector ? min : fill(min, length(initial_x))
    maxes = fill(Inf, length(initial_x))
    o = optimize(td, mins, maxes, initial_x, Fminbox(method), Optim.Options(; kwargs...))
    best_model = set_params(model0, Optim.minimizer(o) .* units, parameters_to_optimize)
    return (best_model, o)
end

