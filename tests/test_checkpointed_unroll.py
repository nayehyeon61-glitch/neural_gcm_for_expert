import jax
import jax.numpy as jnp
import numpy as np

from neuralgcm_expert.training import checkpointed_unroll


class ToyModel:
    """Mimics PressureLevelModel.unroll: nonlinear state map, time-leading outputs."""

    def __init__(self, w):
        self.w = w

    def unroll(self, state, forcings, *, steps, timedelta, start_with_input):
        assert not start_with_input

        def step(s, _):
            s = jnp.tanh(self.w * s + forcings['f'])
            return s, {'x': s * 2.0}

        return jax.lax.scan(step, state, None, length=steps)


def test_matches_plain_unroll_values_and_gradients():
    state = jnp.linspace(-1.0, 1.0, 5)
    forcings = {'f': jnp.full(5, 0.3)}

    def loss(w, fn):
        return jnp.sum(fn(ToyModel(w))['x'] ** 2)

    plain = lambda m: m.unroll(state, forcings, steps=3, timedelta='24h',
                               start_with_input=False)[1]
    remat = lambda m: checkpointed_unroll(m, state, forcings, 3, '24h')
    np.testing.assert_allclose(remat(ToyModel(0.7))['x'], plain(ToyModel(0.7))['x'], rtol=1e-6)
    np.testing.assert_allclose(jax.grad(loss)(0.7, remat), jax.grad(loss)(0.7, plain), rtol=1e-5)
