module MiniKalman

using GaussianDistributions, FillArrays
using GaussianDistributions: dim, logpdf
using StaticArrays
using Statistics, Random, LinearAlgebra

export kfilter, kalman_filter, white_noise1, white_noise2,
    kalman_smoother, kalman_sample, no_noise, log_likelihood, cumulative_log_likelihood

include("utils.jl")
include("filtering.jl")
include("smoothing.jl")
include("kalman_models.jl")

end  # module
