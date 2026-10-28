## Version 2.0.0

ARTIST 2.0.0 reorganizes the package, adds cylindrical-receiver ray tracing and heliostat blocking, and corrects inverse kinematics and its reconstruction workflow.

### What's Changed

- **Breaking:** Reorganized the package into purpose-specific subpackages ([#203](https://github.com/ARTIST-Association/ARTIST/pull/203)):
    - `data_parser/` has moved to `io/`.
    - Ray-tracing functionality now lives in `raytracing/`.
    - NURBS surfaces and helpers now live in `nurbs/`.
    - Reconstruction, optimization, loss, regularization, and training utilities now live in `optim/`.
    - Transformations, coordinate conversions, and rotations now live in `geometry/`.
    - Bitmap and flux-distribution helpers now live in `flux/`.
    - `util/` is now reserved for infrastructure such as constants, tensor-dimension indices, device setup, and distributed setup.
- **Breaking:** Renamed `MotorPositionOptimizer` to `AimPointOptimizer`. It is now available from `artist/optim/aim_point_optimizer.py` ([#214](https://github.com/ARTIST-Association/ARTIST/pull/214)).
- Added ray tracing for cylindrical target areas through a unified target-area model supporting planar and cylindrical receivers ([#197](https://github.com/ARTIST-Association/ARTIST/pull/197)).
- Added support for heliostat blocking, allowing ray tracing to account for heliostats blocking one another's reflected rays ([#187](https://github.com/ARTIST-Association/ARTIST/pull/187), [#197](https://github.com/ARTIST-Association/ARTIST/pull/197)).
- Added a second kinematics reconstruction method and train/test dataset splits for the kinematics and surface reconstructors ([#214](https://github.com/ARTIST-Association/ARTIST/pull/214)).
- Corrected inverse kinematics ([#214](https://github.com/ARTIST-Association/ARTIST/pull/214)).
- Fixed heliostat blocking during batched ray tracing. Rays now use their global heliostat index, making batched and unbatched results consistent ([#187](https://github.com/ARTIST-Association/ARTIST/pull/187)).
- Included additional ray-blocking fixes as part of the cylindrical receiver work ([#197](https://github.com/ARTIST-Association/ARTIST/pull/197)).
- Consolidated test tensors into a single `.pt` fixture file.
- Rebalanced the reconstruction test set to provide better information gain with small sample counts ([#214](https://github.com/ARTIST-Association/ARTIST/pull/214)).

[Full changelog](https://github.com/ARTIST-Association/ARTIST/compare/v1.0.0...v2.0.0)


## Version 1.0.0

### :rocket: **First release** :fire:

### What's Changed
* Features/concentrator by @MarleneBusch in https://github.com/ARTIST-Association/ARTIST/pull/14
* Features/sun rotation by @MarleneBusch in https://github.com/ARTIST-Association/ARTIST/pull/35
* Linear Actuator by @kalebphipps in https://github.com/ARTIST-Association/ARTIST/pull/52
* Heliostat Raytracing with MPI  by @kalebphipps in https://github.com/ARTIST-Association/ARTIST/pull/57
* Features/differentiable nurbs by @MarleneBusch in https://github.com/ARTIST-Association/ARTIST/pull/62
* Create NOTICE by @Markus-Goetz in https://github.com/ARTIST-Association/ARTIST/pull/75
* Maintenance/fair software by @mcw92 in https://github.com/ARTIST-Association/ARTIST/pull/83
* Features/gpu support by @MarleneBusch in https://github.com/ARTIST-Association/ARTIST/pull/94
* Features/alignment optimization by @MarleneBusch in https://github.com/ARTIST-Association/ARTIST/pull/96
* Features/multi heliostats by @MarleneBusch in https://github.com/ARTIST-Association/ARTIST/pull/113
* Features/multiple parallel heliostats by @MarleneBusch in https://github.com/ARTIST-Association/ARTIST/pull/124
* Features/flexible heliostat activation by @MarleneBusch in https://github.com/ARTIST-Association/ARTIST/pull/131
* Features/parallelized heliostat groups by @MarleneBusch in https://github.com/ARTIST-Association/ARTIST/pull/139
* Features/hyperparameter search including motor position optimization by @MarleneBusch in https://github.com/ARTIST-Association/ARTIST/pull/154

## New Contributors
* @MarleneBusch made their first contribution in https://github.com/ARTIST-Association/ARTIST/pull/14
* @kalebphipps made their first contribution in https://github.com/ARTIST-Association/ARTIST/pull/18
* @mcw92 made their first contribution in https://github.com/ARTIST-Association/ARTIST/pull/18
* @Markus-Goetz made their first contribution in https://github.com/ARTIST-Association/ARTIST/pull/75
* @pre-commit-ci[bot] made their first contribution in https://github.com/ARTIST-Association/ARTIST/pull/86
* @Filos1992 made their first contribution in https://github.com/ARTIST-Association/ARTIST/pull/150
