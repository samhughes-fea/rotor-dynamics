# rotor-dynamics

Assembles a rotor. It does not compute blade modes or blade sections.

## Contract

**Input**

- One or more blade modal models, as plain arrays of the kind 1d-fem already produces for a single blade: natural frequencies, mode shapes (`n_dof` by `n_modes`, one mode per column), and mass (a diagonal of generalized modal masses, or the constrained mass matrix in those degrees of freedom).
- A hub and shaft description.
- Rotor speed.
- An aerodynamic load series of the kind a BEM tool returns: sample times and nodal forces on each blade.

**Output**

- The assembled rotor frequencies, including rotation.
- The modal response of the assembled rotor to those loads.

Blade frequencies are data. Centrifugal stiffening and the blade eigenproblem stay in the tool that produced the modes (1d-fem or, later, shell-fem). This package does not import implicit-geometry-kernel, anisotropic-section-model, or shell-fem.

Root-fixed cantilever modes have no hub motion, so the blades do not couple through the hub or the shaft and the gyroscopic matrix is zero. For a non-rotating rotor the assembled frequencies are the blade frequencies, once per blade. The same uncoupled frequencies are returned at any rotor speed, because the speed multiplies that zero gyroscopic matrix. A blade model that already includes rotation keeps the frequencies it was given.

## Install

Python 3.11 or later.

```bash
pip install -e ".[test]"
pytest
```

## Public API

`assemble_rotor(blades, hub_shaft, rotor_speed_rad_s, loads=None)` in `rotor_dynamics`.
