# XLB Execution Flow (Simple)

```mermaid
flowchart TD
    A[User script starts\n(e.g., examples/cfd/*.py)] --> B[Create velocity set\nD2Q9/D3Q19/D3Q27]
    B --> C[xlb.init(...)]
    C --> D[Set global DefaultConfig\nbackend + precision + velocity set]
    D --> E[Create Grid via grid_factory\nJAX or Warp]
    E --> F[Define BCs\n(assign indices/mesh)]
    F --> G[Create IncompressibleNavierStokesStepper]
    G --> H[prepare_fields()\nallocate f_0, f_1, bc_mask, missing_mask]
    H --> I[Initialize f_0 (equilibrium/custom)]
    I --> J[Build boundary masks\n+ optional BC auxiliary data]

    J --> K{Time-step loop}
    K --> L[Streaming]
    L --> M[Apply streaming-stage BCs]
    M --> N[Compute macroscopic fields\n(rho, u)]
    N --> O[Compute equilibrium feq]
    O --> P[Collision (BGK/KBC/...)]
    P --> Q[Apply collision-stage BCs\n+ aux update]
    Q --> R[Write output buffer\nand swap f_0/f_1]
    R --> K

    K --> S[Periodic post-processing\n(VTK/images/macroscopic output)]
    S --> T[Simulation end]
```

## Notes
- The same high-level pipeline is used for both backends; only kernel implementation differs (JAX vs Warp).
- In JAX, operators are JIT-compiled and dispatched through the `Operator` backend registry.
- In Warp, kernels are built/launched through Warp functionals/kernels.
