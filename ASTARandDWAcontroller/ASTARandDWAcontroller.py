from controller import Robot
import numpy as np
import math

# =========================
# Robot Setup
# =========================
robot = Robot()
timestep = int(robot.getBasicTimeStep())
dt = timestep / 1000.0

LEFT_MOTOR_NAME = "left wheel"
RIGHT_MOTOR_NAME = "right wheel"
LEFT_SENSOR_NAME = "left wheel sensor"
RIGHT_SENSOR_NAME = "right wheel sensor"
LIDAR_NAME = "lidar"

left_motor = robot.getDevice(LEFT_MOTOR_NAME)
right_motor = robot.getDevice(RIGHT_MOTOR_NAME)
left_sensor = robot.getDevice(LEFT_SENSOR_NAME)
right_sensor = robot.getDevice(RIGHT_SENSOR_NAME)
lidar = robot.getDevice(LIDAR_NAME)

if left_motor is None or right_motor is None:
    print("Motor device not found.")
    exit()

if left_sensor is None or right_sensor is None:
    print("Wheel sensor not found.")
    exit()

if lidar is None:
    print("Lidar not found.")
    exit()

left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

left_sensor.enable(timestep)
right_sensor.enable(timestep)
lidar.enable(timestep)

robot.step(timestep)

prev_left = left_sensor.getValue()
prev_right = right_sensor.getValue()

if math.isnan(prev_left) or math.isnan(prev_right):
    print("Encoder values are NaN.")
    exit()

# =========================
# Load A* Waypoints
# =========================
path = np.load("astar_waypoints.npy")

if len(path) == 0:
    print("astar_waypoints.npy is empty.")
    exit()

print("Loaded waypoints:", path[:5], "...")

# =========================
# Robot Parameters
# =========================
WHEEL_RADIUS = 0.0975
AXLE_LENGTH = 0.33
MAX_SPEED = 4.0

x = 2.0
y = 1.0
theta = -1.5708

WAYPOINT_TOL = 0.08
FINAL_WAYPOINT_TOL = 0.10
current_wp = 0

current_v = 0.0
current_w = 0.0

# =========================
# DWA Parameters
# =========================
MAX_V = 0.20
MIN_V = 0.0
MAX_W = 1.0

MAX_ACC_V = 0.5
MAX_ACC_W = 1.8

PREDICT_TIME = 1.0
V_SAMPLES = 4
W_SAMPLES = 7

ROBOT_RADIUS = 0.25

HEADING_WEIGHT = 2.5
CLEARANCE_WEIGHT = 2.0
VELOCITY_WEIGHT = 0.3
ALIGNMENT_WEIGHT = 0.8

# =========================
# LiDAR Parameters
# =========================
LIDAR_SECTOR_LIMIT_DEG = 120
LIDAR_YAW_OFFSET = 0.0
LIDAR_BEAM_STEP = 4

# =========================
# Debug / Control
# =========================
step_count = 0
CONTROL_SKIP = 2

last_v = 0.0
last_w = 0.0

# =========================
# Trajectory Logging
# =========================
trajectory = []
time_log = []

# =========================
# Helper Functions
# =========================
def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def simulate_trajectory(x, y, theta, v, w, dt, predict_time):
    traj = []
    px, py, ptheta = x, y, theta
    steps = int(predict_time / dt)

    for _ in range(steps):
        px += v * math.cos(ptheta) * dt
        py += v * math.sin(ptheta) * dt
        ptheta = normalize_angle(ptheta + w * dt)
        traj.append((px, py, ptheta))

    return traj


def lidar_to_world_obstacles(ranges, robot_x, robot_y, robot_theta, fov, max_range):
    obstacles = []
    n = len(ranges)

    if n < 2:
        return obstacles

    sector_limit = math.radians(LIDAR_SECTOR_LIMIT_DEG)

    for i in range(0, n, LIDAR_BEAM_STEP):
        r = ranges[i]

        if math.isinf(r) or r <= 0.0 or r > max_range:
            continue

        beam_angle = -fov / 2.0 + i * (fov / (n - 1)) + LIDAR_YAW_OFFSET

        if abs(beam_angle) > sector_limit:
            continue

        ox_r = r * math.cos(beam_angle)
        oy_r = r * math.sin(beam_angle)

        ox_w = robot_x + ox_r * math.cos(robot_theta) - oy_r * math.sin(robot_theta)
        oy_w = robot_y + ox_r * math.sin(robot_theta) + oy_r * math.cos(robot_theta)

        obstacles.append((ox_w, oy_w))

    return obstacles


def min_obstacle_distance(traj, obstacles):
    if len(obstacles) == 0:
        return 5.0

    min_dist = float("inf")

    for tx, ty, _ in traj:
        for ox, oy in obstacles:
            d = math.hypot(ox - tx, oy - ty)
            if d < min_dist:
                min_dist = d

    return min_dist


def is_collision(traj, obstacles, robot_radius):
    if len(obstacles) == 0:
        return False

    for tx, ty, _ in traj:
        for ox, oy in obstacles:
            d = math.hypot(ox - tx, oy - ty)
            if d <= robot_radius:
                return True

    return False


def trajectory_score(traj, goal_x, goal_y, obstacles, v):
    end_x, end_y, end_theta = traj[-1]

    dist_to_goal = math.hypot(goal_x - end_x, goal_y - end_y)
    heading_score = -dist_to_goal

    clearance = min_obstacle_distance(traj, obstacles)
    clearance_score = min(clearance, 2.0)

    velocity_score = v

    goal_heading = math.atan2(goal_y - end_y, goal_x - end_x)
    heading_error = abs(normalize_angle(goal_heading - end_theta))
    alignment_score = -heading_error

    score = (
        HEADING_WEIGHT * heading_score +
        CLEARANCE_WEIGHT * clearance_score +
        VELOCITY_WEIGHT * velocity_score +
        ALIGNMENT_WEIGHT * alignment_score
    )
    return score


def dwa_control(x, y, theta, goal_x, goal_y, obstacles, current_v, current_w):
    v_min = max(MIN_V, current_v - MAX_ACC_V * dt)
    v_max = min(MAX_V, current_v + MAX_ACC_V * dt)

    w_min = max(-MAX_W, current_w - MAX_ACC_W * dt)
    w_max = min(MAX_W, current_w + MAX_ACC_W * dt)

    if abs(v_max - v_min) < 1e-6:
        v_min = max(MIN_V, current_v - 0.02)
        v_max = min(MAX_V, current_v + 0.02)

    if abs(w_max - w_min) < 1e-6:
        w_min = max(-MAX_W, current_w - 0.05)
        w_max = min(MAX_W, current_w + 0.05)

    v_candidates = np.linspace(v_min, v_max, V_SAMPLES)
    w_candidates = np.linspace(w_min, w_max, W_SAMPLES)

    best_score = -float("inf")
    best_v = 0.0
    best_w = 0.0
    best_traj = None

    for v in v_candidates:
        for w in w_candidates:
            traj = simulate_trajectory(x, y, theta, v, w, dt, PREDICT_TIME)

            if not traj:
                continue

            if is_collision(traj, obstacles, ROBOT_RADIUS):
                continue

            score = trajectory_score(traj, goal_x, goal_y, obstacles, v)

            if score > best_score:
                best_score = score
                best_v = v
                best_w = w
                best_traj = traj

    if best_traj is None:
        desired_heading = math.atan2(goal_y - y, goal_x - x)
        angle_error = normalize_angle(desired_heading - theta)
        best_v = 0.0
        best_w = max(-MAX_W, min(MAX_W, 1.2 * angle_error))

    return best_v, best_w


def vw_to_wheel_speeds(v, w):
    left_speed = (v - (AXLE_LENGTH / 2.0) * w) / WHEEL_RADIUS
    right_speed = (v + (AXLE_LENGTH / 2.0) * w) / WHEEL_RADIUS

    left_speed = max(-MAX_SPEED, min(MAX_SPEED, left_speed))
    right_speed = max(-MAX_SPEED, min(MAX_SPEED, right_speed))

    return left_speed, right_speed


# =========================
# Main Control Loop
# =========================
theta = normalize_angle(theta)

while robot.step(timestep) != -1:
    step_count += 1

    left_now = left_sensor.getValue()
    right_now = right_sensor.getValue()

    if math.isnan(left_now) or math.isnan(right_now):
        print("Still receiving NaN from wheel sensors.")
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        continue

    # Odometry update
    dleft = left_now - prev_left
    dright = right_now - prev_right

    prev_left = left_now
    prev_right = right_now

    dl = dleft * WHEEL_RADIUS
    dr = dright * WHEEL_RADIUS

    dc = (dl + dr) / 2.0
    dtheta = (dr - dl) / AXLE_LENGTH

    theta_mid = theta + dtheta / 2.0
    x += dc * math.cos(theta_mid)
    y += dc * math.sin(theta_mid)
    theta = normalize_angle(theta + dtheta)

    # Log actual trajectory
    trajectory.append((x, y))
    time_log.append(step_count * dt)

    if current_wp >= len(path):
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)

        trajectory_np = np.array(trajectory)
        np.save("actual_trajectory.npy", trajectory_np)
        np.savetxt(
            "actual_trajectory.csv",
            trajectory_np,
            delimiter=",",
            header="x,y",
            comments=""
        )

        print("Path completed.")
        print("Saved actual trajectory to actual_trajectory.npy and actual_trajectory.csv")
        break

    target_x, target_y = path[current_wp]

    dx = target_x - x
    dy = target_y - y
    distance = math.hypot(dx, dy)

    wp_tol = FINAL_WAYPOINT_TOL if current_wp == len(path) - 1 else WAYPOINT_TOL

    if distance < wp_tol:
        print(f"Reached waypoint {current_wp}: ({target_x:.2f}, {target_y:.2f})")
        current_wp += 1
        continue

    if step_count % CONTROL_SKIP == 0:
        ranges = lidar.getRangeImage()
        obstacles = lidar_to_world_obstacles(
            ranges,
            x, y, theta,
            lidar.getFov(),
            lidar.getMaxRange()
        )

        v, w = dwa_control(
            x, y, theta,
            target_x, target_y,
            obstacles,
            current_v, current_w
        )

        last_v = v
        last_w = w
        current_v = v
        current_w = w
    else:
        v = last_v
        w = last_w
        obstacles = []

    left_speed, right_speed = vw_to_wheel_speeds(v, w)

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)

    if step_count % 10 == 0:
        angle_to_target = math.atan2(dy, dx)
        angle_error = normalize_angle(angle_to_target - theta)

        if step_count % CONTROL_SKIP == 0 and len(obstacles) > 0:
            nearest = min(math.hypot(ox - x, oy - y) for ox, oy in obstacles)
        else:
            nearest = float("inf")

        print(
            f"x={x:.2f}, y={y:.2f}, theta={math.degrees(theta):.1f} deg, "
            f"wp={current_wp}, target=({target_x:.2f},{target_y:.2f}), "
            f"dist={distance:.2f}, err={math.degrees(angle_error):.1f}, "
            f"nearest_obs={nearest:.2f}, "
            f"v={v:.2f}, w={w:.2f}, "
            f"left={left_speed:.2f}, right={right_speed:.2f}"
        )