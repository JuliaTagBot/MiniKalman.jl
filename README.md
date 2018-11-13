A Kalman Filtering package. Documentation will be added eventually.

Short notes:

- Models are specified through writing a structure that inherits from `MiniKalman.Model`, and which implements (some of) these functions: `transition_mat, transition_noise, observation_mat, observation_noise, observation, initial_state`. These last two are optional, and can be passed as keyword arguments to `kalman_filter`.
- For example, `transition_mat(model::MyModel, inputs, t::Int)` should return the
transition matrix at time `t` under the given `model`'s parameters. `inputs` is an
arbitrary object that is passed to `kalman_filter`, and is passed through without
modification. It can be, for instance, a vector of forcing terms `forcing_terms` (you
would then use `forcing_terms[t]`), or even a DataFrame (though that would be a
performance concern - mind your type stability).
- For an example, see the [tests](test/runtests.jl)
- This code was only tested with `SVector`/`SMatrix`. Might have problems with plain arrays.
- TODO: I occasionally use `state` to refer either to a probability distribution, or
  to a sample. It should be more consistent.