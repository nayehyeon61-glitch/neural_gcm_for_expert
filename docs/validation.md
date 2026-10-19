# Validation record — 2026-10-19

Environment: CPU, Python 3.12, NeuralGCM 1.2.2, Dinosaur 1.5.0,
JAX/JAXlib 0.7.2, Haiku 0.0.17, Optax 0.2.8. Editable package build and CLI
entrypoint were checked. `pip check` reported no broken requirements.

Final standard suite (including direct upstream metric parity): **51 passed,
1 skipped** in 33.64 seconds. The skipped opt-in full-gradient test was executed
separately and passed as described below. Warnings come from the upstream toy
checkpoint's legacy pickled JAX arrays.

## What was executed

- Independent temperature/momentum/moisture/residual network tests: exact output
  shapes and None masks, deterministic apply, component sum, full-input coupling,
  bounded residual and independent parameter scopes.
- Native primitive-equation tests using official modules: explicit + implicit +
  learned tendencies, custom-coordinate interpolation, clock derivative, JVP,
  and parameter gradient through acceleration.
- Config/checkpoint tests: no-memory/no-randomness conversion, preserved official
  stepper, original tower factory and inherited Gin scopes, correct state-encoder
  weight transfer rather than memory-encoder transfer, checkpoint restore.
- Data/evaluation tests: positive forecast leads, exact coordinate/time alignment,
  forcing persistence, pooled squared-error aggregation, NetCDF/JSON export.
  Direct numerical parity with the upstream reference `metrics.RMSE` was tested
  with nonuniform errors on a Gaussian grid. This optional parity test uses a
  local upstream checkout supplied by `NEURALGCM_REFERENCE_DIR`.
- Loss tests: analytic coupled ODE with nonzero cross derivatives, forcing/time
  dependence, masks and scales, finite differences, and named native state fields.

## Real official-solver gradient smoke checks

The included fixture explicitly converts the **bundled stochastic toy** into a
deterministic memoryless experiment, reduces widths and grids, and uses the one
bundled ERA5 snapshot to construct a synthetic persistence target. It does not
evaluate the forecast quality of an official pretrained deterministic model.

Forecast-only run: **passed**, 91.59 seconds CPU.

Latest production conversion path with **both positive tendency and acceleration
loss weights**: **passed**, 146.67 seconds CPU. The check performs:

1. Actual official pressure-level encoding with `memory=None`.
2. Actual official IMEX rollout and decoding.
3. Gradient through rollout, native RHS, and JVP acceleration.
4. Accepted optimizer update with changed expert weights and exactly unchanged
   frozen encoder/decoder/dynamics parameters.
5. Checkpoint save/restore and matching public output shapes and numerical output.

Reproduce the opt-in test:

```bash
RUN_NEURALGCM_INTEGRATION=1 RUN_NEURALGCM_NATIVE_LOSSES=1 \
  python -m pytest -q tests/test_training.py
```

Or run the standalone smoke script:

```bash
python -m examples.smoke_test --native-losses --checkpoint /tmp/ngcm-expert-smoke.pkl
```

The official v1.2.2 deterministic 2.8° paper Gin configuration was additionally
parsed in all three modes: baseline retains native memory; memoryless and experts
remove it; all are deterministic; experts inherit the original `EpdTower`.

## Not established by these checks

No full-resolution deterministic checkpoint training, multi-year ERA5 experiment,
GPU training, forecast-skill improvement, physical-process identifiability,
conservation guarantee, or climate-timescale stability has been demonstrated.
Those require data, compute and a matched baseline experiment. CPU timings above
are compilation/test runtimes, not production performance estimates.
