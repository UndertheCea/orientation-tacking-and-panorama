# ECE276A Project 1: Orientation Tracking & Panorama Construction

This project estimates body orientation from IMU measurements using quaternion kinematics and projected gradient descent optimization, then constructs panoramic images from camera data.

## Requirements

- **Python Version**: 3.9.6 (recommended)

## Dependencies

Install the required packages using `pip`:

```bash
pip install numpy scipy matplotlib transforms3d torch pillow
```

## Project Structure

```
ECE276A_PR1/
├── README.md                              # This file
├── code/
│   ├── imu_quaternion_integration_vs_vicon.ipynb  # Main notebook
│   ├── load_data.py                       # Data loading utilities
│   └── rotplot.py                         # Rotation plotting utilities
│
├── data/
│   ├── trainset/                          # Training datasets (1-9)
│   │   ├── imu/                           # IMU raw measurements
│   │   │   ├── imuRaw1.p
│   │   │   ├── imuRaw2.p
│   │   │   └── ... (imuRaw1 through imuRaw9)
│   │   ├── cam/                           # Camera images (datasets 1,2,8,9 only)
│   │   │   ├── cam1.p
│   │   │   ├── cam2.p
│   │   │   ├── cam8.p
│   │   │   └── cam9.p
│   │   └── vicon/                         # Ground truth rotations (all 1-9)
│   │       ├── viconRot1.p
│   │       ├── viconRot2.p
│   │       └── ... (viconRot1 through viconRot9)
│   │
│   └── testset/                           # Test datasets (10-11, no VICON)
│       ├── imu/
│       │   ├── imuRaw10.p
│       │   └── imuRaw11.p
│       └── cam/
│           ├── cam10.p
│           └── cam11.p
│
├── docs/                                  # Documentation folder
│
└── results/                               # Output results (auto-generated)
    ├── run_YYYYMMDD_HHMMSS/
    │   ├── dataset_1_q_optimized.pt       # PyTorch optimized quaternions
    │   ├── dataset_1_optimized_euler.png  # Euler angles plot
    │   ├── dataset_1_convergence.png      # Convergence curve
    │   ├── dataset_1_panorama.png         # Panoramic image
    │   ├── ... (all 11 datasets)
    │   ├── subplot_optimized_euler_3x3.png    # Training Euler subplots
    │   ├── subplot_convergence_3x3.png        # Training convergence subplots
    │   ├── subplot_panorama_2x2.png           # Training panorama subplots
    │   ├── testset_subplots/
    │   │   ├── testset_optimized_euler_1x2.png    # Testset Euler (1×2)
    │   │   ├── testset_convergence_1x2.png        # Testset convergence (1×2)
    │   │   └── testset_panorama_1x2.png           # Testset panorama (1×2)
    │   └── summary_results.txt             # Summary metrics
    │
    └── ... (older runs)
```

## Running the Code

### Main Notebook
Open and run the Jupyter notebook:
```bash
cd code
jupyter notebook imu_quaternion_integration_vs_vicon.ipynb
```

The notebook performs:
1. **Data Loading & Calibration** — Loads IMU/VICON/camera data, calibrates using static period
2. **Quaternion Integration** — Integrates gyro measurements to estimate orientation
3. **Observation Model** — Validates gravity measurements against estimated orientation
4. **Optimization** — Refines orientation trajectory using projected gradient descent
5. **Panorama Construction** — Stitches camera images using optimized orientations
6. **Visualization** — Generates subplots and saves results

## Key Algorithms

### Motion Model
Quaternion kinematics from gyro angular velocity:
$$q_{t+1} = q_t \circ \exp\left([0, \frac{\tau_t \omega_t}{2}]\right)$$

### Observation Model
Gravity vector rotated into IMU frame:
$$h(q_t) = q_t^{-1} \circ [0,0,0,1] \circ q_t$$

### Optimization
Minimize weighted cost:
$$\mathcal{c}(q_{1:T}) = 1.9 \cdot \text{motion\_cost} + 0.1 \cdot \text{obs\_cost}$$
Subject to: $\|q_t\|_2 = 1$ for all $t$ (projected gradient descent)

### Panorama Projection
Three-step transformation:
1. **Sphere Projection** — Map camera pixels to unit sphere
2. **World Frame Rotation** — Apply estimated body orientation
3. **Cylindrical Unwrap** — Project onto panorama canvas

## Output Files

For each dataset, generates:
- `dataset_X_q_optimized.pt` — PyTorch tensor of optimized quaternions
- `dataset_X_optimized_euler.png` — Roll/pitch/yaw vs VICON (trainset) or standalone (testset)
- `dataset_X_convergence.png` — Cost vs iteration (log scale)
- `dataset_X_panorama.png` — Stitched panoramic image
- `summary_results.txt` — Metrics summary (cost reduction, iterations, coverage %)

## Configuration Parameters

Key tunable parameters in the notebook:
- `learning_rate` — Adam optimizer learning rate (default: 0.01)
- `num_iterations` — Max optimization iterations (default: 100 trainset, 120 testset)
- `lr_decay` — LR decay factor on plateau (default: 0.5)
- `lr_patience` — Iterations before LR decay (default: 7)
- `min_lr` — Minimum learning rate floor (default: 1e-4)
- `early_stop_pct` — Relative improvement threshold for early stopping (default: 1e-3 = 0.1%)
- `PANO_RESOLUTION` — Panorama pixels per radian (default: 100)

## Data Format

All data files use Python pickle format (`.p`):
- **IMU** — (7, N) array: [timestamps, ax, ay, az, wx, wy, wz]
- **VICON** — Dict with `'rots'` (3×3×N) and `'ts'` (N,)
- **Camera** — Dict with `'cam'` (240×320×3×M) and `'ts'` (M,)

Load via: `data = read_data("file.p")`

## Notes

- Trainset datasets 1-9 include VICON ground truth for comparison
- Testset datasets 10-11 contain only IMU and camera (no ground truth)
- Only camera data available for datasets 1, 2, 8, 9 (trainset) and 10, 11 (testset)
- Results are timestamped and stored in `results/run_YYYYMMDD_HHMMSS/`

## Author

ECE276A Project 1 — Spring 2026

