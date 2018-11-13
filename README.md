Some notes:

- Models are specified through writing a structure that inherits from `MiniKalman.Model`, and which implements (some of) these functions: `transition_mat, transition_noise, observation_mat, observation_noise, observation, initial_state`. These last two are optional, and can be passed as keyword arguments to `kalman_filter`.
- For an example, see the [tests](test/runtests.jl)
- This code was only tested with `SVector`/`SMatrix`. Might have problems with plain arrays.
- TODO: I occasionally use `state` to refer either to a probability distribution, or
  to a sample. It should be more consistent.