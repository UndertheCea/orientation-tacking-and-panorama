import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
import transforms3d as t3d
from load_data import read_data, tic, toc
import sys

# ============================================================================
# QUATERNION HELPER FUNCTIONS
# ============================================================================

def quaternion_multiply(q1, q2):
    """
    Quaternion multiplication: q1 ⊙ q2
    q1, q2: quaternions as [w, x, y, z] (scalar first)
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])

def quaternion_conjugate(q):
    """
    Quaternion conjugate: q* = [w, -x, -y, -z]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quaternion_exponential(v):
    """
    Quaternion exponential of a pure quaternion [0, x, y, z]
    exp([0, v]) = [cos(||v||/2), sin(||v||/2) * v/||v||]
    v: 3D vector [x, y, z]
    """
    norm_v = np.linalg.norm(v)
    
    if norm_v < 1e-10:  # avoid division by zero
        return np.array([1.0, v[0]/2, v[1]/2, v[2]/2])
    
    w = np.cos(norm_v / 2)
    xyz = np.sin(norm_v / 2) * v / norm_v
    
    return np.array([w, xyz[0], xyz[1], xyz[2]])

def quaternion_to_euler(q):
    """
    Convert quaternion [w, x, y, z] to Euler angles (roll, pitch, yaw) in radians
    """
    # Use transforms3d to convert
    # First convert to rotation matrix
    quat_xyzw = np.array([q[1], q[2], q[3], q[0]])  # transforms3d uses [x, y, z, w]
    roll, pitch, yaw = t3d.euler.quat2euler(quat_xyzw)
    
    return roll, pitch, yaw

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion [w, x, y, z]
    """
    quat_xyzw = t3d.euler.euler2quat(roll, pitch, yaw)
    q = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # to [w, x, y, z]
    return q

def normalize_quaternion(q):
    """
    Normalize quaternion to unit length
    """
    return q / np.linalg.norm(q)

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading data...")
ts = tic()

dataset = "1"
cfile = "../data/trainset/cam/cam" + dataset + ".p"
ifile = "../data/trainset/imu/imuRaw" + dataset + ".p"
vfile = "../data/trainset/vicon/viconRot" + dataset + ".p"

camd = read_data(cfile)
imud = read_data(ifile)
vicd = read_data(vfile)

toc(ts, "Data import")

# Extract IMU data
# imud is typically a dict with keys: 'vals' (6D: [ax, ay, az, wx, wy, wz]), 'ts' (timestamps)
imu_vals = imud['vals']  # shape: (6, N) - [accel_x, accel_y, accel_z, omega_x, omega_y, omega_z]
imu_ts = imud['ts']      # shape: (1, N) or (N,)

# Ensure timestamps are 1D
if imu_ts.ndim > 1:
    imu_ts = imu_ts.flatten()

# Extract angular velocity and acceleration
accel = imu_vals[:3, :]  # shape: (3, N)
omega = imu_vals[3:, :]  # shape: (3, N) - angular velocity in rad/sec

# Extract VICON data (rotation matrices)
# vicd typically has 'rots' (3D rotation matrices) and 'ts' (timestamps)
vicon_rots = vicd['rots']  # shape: (3, 3, N)
vicon_ts = vicd['ts']      # shape: (1, N) or (N,)
if vicon_ts.ndim > 1:
    vicon_ts = vicon_ts.flatten()

print(f"IMU data shape: {imu_vals.shape}, timestamps: {imu_ts.shape}")
print(f"VICON data shape: {vicon_rots.shape}, timestamps: {vicon_ts.shape}")

# ============================================================================
# QUATERNION INTEGRATION
# ============================================================================

print("\nIntegrating angular velocity to estimate orientation...")
ts = tic()

N = imu_vals.shape[1]

# Initialize quaternion: q0 = [1, 0, 0, 0] (identity/no rotation)
q_est = np.zeros((4, N))
q_est[:, 0] = np.array([1.0, 0.0, 0.0, 0.0])

# Integrate using quaternion kinematics: qt+1 = qt ⊙ exp([0, τt*ωt/2])
for i in range(N - 1):
    tau = imu_ts[i + 1] - imu_ts[i]  # time difference
    omega_t = omega[:, i]             # angular velocity at time t
    
    # Quaternion kinematics: qt+1 = qt ⊙ exp([0, τ*ω/2])
    exp_quat = quaternion_exponential(tau * omega_t / 2.0)
    q_est[:, i + 1] = quaternion_multiply(q_est[:, i], exp_quat)
    
    # Normalize to ensure unit quaternion
    q_est[:, i + 1] = normalize_quaternion(q_est[:, i + 1])

toc(ts, "Quaternion integration")

# ============================================================================
# CONVERT TO EULER ANGLES
# ============================================================================

print("Converting quaternions to Euler angles...")
ts = tic()

euler_est = np.zeros((3, N))  # roll, pitch, yaw

for i in range(N):
    roll, pitch, yaw = quaternion_to_euler(q_est[:, i])
    euler_est[:, i] = [roll, pitch, yaw]

toc(ts, "Euler angle conversion")

# Convert VICON rotation matrices to Euler angles
N_vicon = vicon_rots.shape[2]
euler_vicon = np.zeros((3, N_vicon))

for i in range(N_vicon):
    # transforms3d expects rotation matrix in specific format
    roll, pitch, yaw = t3d.euler.mat2euler(vicon_rots[:, :, i])
    euler_vicon[:, i] = [roll, pitch, yaw]

# ============================================================================
# PLOTTING
# ============================================================================

print("\nPlotting results...")

# Convert timestamps to relative time (seconds from start)
imu_time = imu_ts - imu_ts[0]
vicon_time = vicon_ts - vicon_ts[0]

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Roll
axes[0].plot(imu_time, np.degrees(euler_est[0, :]), 'b-', label='Estimated (IMU integration)', linewidth=2)
axes[0].plot(vicon_time, np.degrees(euler_vicon[0, :]), 'r--', label='VICON ground truth', linewidth=2)
axes[0].set_ylabel('Roll (degrees)', fontsize=12)
axes[0].set_title('Orientation Estimation: Quaternion Integration vs VICON', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Pitch
axes[1].plot(imu_time, np.degrees(euler_est[1, :]), 'b-', label='Estimated (IMU integration)', linewidth=2)
axes[1].plot(vicon_time, np.degrees(euler_vicon[1, :]), 'r--', label='VICON ground truth', linewidth=2)
axes[1].set_ylabel('Pitch (degrees)', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

# Yaw
axes[2].plot(imu_time, np.degrees(euler_est[2, :]), 'b-', label='Estimated (IMU integration)', linewidth=2)
axes[2].plot(vicon_time, np.degrees(euler_vicon[2, :]), 'r--', label='VICON ground truth', linewidth=2)
axes[2].set_xlabel('Time (seconds)', fontsize=12)
axes[2].set_ylabel('Yaw (degrees)', fontsize=12)
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('orientation_comparison.png', dpi=150, bbox_inches='tight')
print("Plot saved as 'orientation_comparison.png'")
plt.show()

# ============================================================================
# COMPUTE ERROR METRICS
# ============================================================================

print("\n" + "="*60)
print("ERROR ANALYSIS")
print("="*60)

# Interpolate estimated angles to VICON timestamps for fair comparison
from scipy.interpolate import interp1d

# Interpolate roll, pitch, yaw
f_roll = interp1d(imu_time, euler_est[0, :], kind='linear', fill_value='extrapolate')
f_pitch = interp1d(imu_time, euler_est[1, :], kind='linear', fill_value='extrapolate')
f_yaw = interp1d(imu_time, euler_est[2, :], kind='linear', fill_value='extrapolate')

euler_est_interp = np.array([
    f_roll(vicon_time),
    f_pitch(vicon_time),
    f_yaw(vicon_time)
])

# Compute errors
error_roll = euler_est_interp[0, :] - euler_vicon[0, :]
error_pitch = euler_est_interp[1, :] - euler_vicon[1, :]
error_yaw = euler_est_interp[2, :] - euler_vicon[2, :]

# Convert to degrees
error_roll_deg = np.degrees(error_roll)
error_pitch_deg = np.degrees(error_pitch)
error_yaw_deg = np.degrees(error_yaw)

print(f"\nRoll Error:")
print(f"  Mean: {np.mean(error_roll_deg):.4f}°")
print(f"  Std Dev: {np.std(error_roll_deg):.4f}°")
print(f"  Max: {np.max(np.abs(error_roll_deg)):.4f}°")

print(f"\nPitch Error:")
print(f"  Mean: {np.mean(error_pitch_deg):.4f}°")
print(f"  Std Dev: {np.std(error_pitch_deg):.4f}°")
print(f"  Max: {np.max(np.abs(error_pitch_deg)):.4f}°")

print(f"\nYaw Error:")
print(f"  Mean: {np.mean(error_yaw_deg):.4f}°")
print(f"  Std Dev: {np.std(error_yaw_deg):.4f}°")
print(f"  Max: {np.max(np.abs(error_yaw_deg)):.4f}°")

print("\n" + "="*60)
print("If errors are small (< 5°), IMU calibration is good!")
print("="*60)
