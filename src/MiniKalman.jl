module MiniKalman

using GaussianDistributions, FillArrays
using GaussianDistributions: dim, logpdf
using StaticArrays
using Statistics, Random, LinearAlgebra
using Unitful: ustrip, unit   # could be @required ?
using QuickTypes: roottypeof  # TODO: get rid of this dependency
using Random: GLOBAL_RNG
using DocStringExtensions
@template (FUNCTIONS, METHODS) =
    """
    $(SIGNATURES)
    $(DOCSTRING)
    """

export kfilter, kalman_filter, white_noise1, white_noise2,
    kalman_smoother, kalman_sample, no_noise, log_likelihood, cumulative_log_likelihood,
    sample_and_recover, optimize, marginal, kalman_predicted, kalman_filtered


include("utils.jl")
include("filtering.jl")
include("smoothing.jl")
include("optim.jl")
include("sampling.jl")

end  # module
