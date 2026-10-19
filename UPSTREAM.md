# Upstream provenance

This repository extends the official `neuralgcm==1.2.2` Python package; it does
not copy or replace the dynamics core with a surrogate. The corresponding
upstream release is tag `v1.2.2`, commit
`c91d2007ca37a4c0842f3ff83256d32e1ec30815`:

https://github.com/neuralgcm/neuralgcm/tree/v1.2.2

The upstream `main` branch was also inspected at
`bd23cd0ac41213b8ade598415e152af4068091c5`. Its evolving APIs are not the
compatibility target of this experiment. Pinned versions are in pyproject.toml.

NeuralGCM and Dinosaur code are Apache-2.0 licensed. The official pretrained
weights have the separate CC BY-SA 4.0 license; they are not redistributed here.
See https://neuralgcm.readthedocs.io/en/latest/checkpoints.html.

Official implementations reused at runtime:

- `PressureLevelModel`: ERA5 pressure-level encode, decode, unroll and xarray I/O.
- `DivCurlNeuralParameterization`: features, tendency units, div/curl conversion,
  prediction mask and filtering.
- `DycoreWithPhysicsCorrector` and `CustomCoordsCorrector`: physical equations,
  interpolation, original IMEX integration and numerical filtering.
- `ColumnTower` / checkpoint's configured tower: each independent expert network.
- `horizontal.integrate`: the same spherical quadrature reduction used by the
  upstream reference metric `nodal_surface_mean`.

The upstream wheel does not ship `legacy/reference_code`. This repository's
evaluation computes its deterministic RMSE reduction with the same underlying
quadrature; it does not claim to reproduce the entire paper/WeatherBench2
benchmark or its datasets merely by running a smoke test.
