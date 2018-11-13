using Test
using MiniKalman, GaussianDistributions, StaticArrays, FillArrays, Statistics, Random,
    Unitful, Parameters

@with_kw struct RandomWalk <: MiniKalman.Model
    step_size
    obs_noise
end
MiniKalman.observation_noise(rw::RandomWalk, _, _) = white_noise1(rw.obs_noise)
MiniKalman.transition_noise(rw::RandomWalk, _, _) = white_noise1(rw.step_size)
MiniKalman.initial_state(rw::RandomWalk) = white_noise1(0.5) # centered around 0

# Check that sampling from the true model, then inferring the parameters of RandomWalk
# through maximum likelihood applied to that sample, yields the true model.
true_model = RandomWalk(step_size=2.0, obs_noise=0.5)
sr = sample_and_recover(true_model, nothing, 100000, 
                        start_model=RandomWalk(step_size=1.0, obs_noise=1.0),
                        rng=MersenneTwister(1))

@test sr.estimated_model.step_size ≈ sr.true_model.step_size rtol=0.02
@test sr.estimated_model.obs_noise ≈ sr.true_model.obs_noise rtol=0.02



# Needs revision:
# N = 100
# rng = MersenneTwister(1)

# # Need to generate it as regular vectors/matrices first to use `\` below.
# xx = rand(rng, 1, 1,N)
# XX = hcat(ones(1,1,N), xx)
# XX_mat = [convert(SMatrix{1,2}, XX[:,:,i]) for i in 1:N]
# YY = 7*xx[:] + randn(rng, N) .* sqrt(1) .+ 15
# YY_vec = map(SVector, YY)

# xxf2, ll2 = kalman_filter(Gaussian([20., 20.0], [1000.0 0; 0 1000.0]), YY_vec,
#                           Fill(MiniKalman.white_noise1(10.0), N),
#                           observation_mats=XX_mat)

# xf = mean(xxf2[end])
# @test xf ≈ XX[1, :, :]' \ YY rtol=0.01
# @assert N==100 # otherwise the following test is not exact
# @test xf ≈ [15.0589, 6.73069] rtol=0.001

# @test sum(ll2) ≈ -328.39288090529027
# rand(white_noise2(1.0u"m^2"))
