using Base.Test
using MiniKalman, GaussianDistributions, StaticArrays, FillArrays

N = 100
rng = MersenneTwister(1)

# Need to generate it as regular vectors/matrices first to use `\` below.
xx = rand(rng, 1, 1,N)
XX = hcat(ones(1,1,N), xx)
XX_mat = [convert(SMatrix{1,2}, XX[:,:,i]) for i in 1:N]
YY = 7*xx[:] + randn(rng, N)*sqrt(1)+15
YY_vec = map(SVector, YY)

xxf2, ll2 = kalman_filter(Gaussian([20., 20.0], eye(2)*1000), YY_vec,
                          Fill(MiniKalman.white_noise1(10.0), N),
                          observation_mats=XX_mat)
ll2

xf = mean(xxf2[end])
@test xf ≈ XX[1, :, :]' \ YY rtol=0.01
@assert N==100 # otherwise the following test is not exact
@test xf ≈ [15.0589, 6.73069] rtol=0.001

@test sum(ll2) ≈ -328.39288090529027

