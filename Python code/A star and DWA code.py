import cv2
import numpy as np
import matplotlib.pyplot as plt
import heapq
import math


# STEP 1: Load Map

img = cv2.imread(r"C:\Users\Hari\Desktop\Mobile robotics\.Roommap-filtered.jpg")

if img is None:
    raise FileNotFoundError("Map image not found!")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(6, 8))
plt.imshow(gray, cmap='gray')
plt.title("Original Map")
plt.axis("off")
plt.show()


# STEP 2: Binary Conversion
# White = free space
# Black = obstacle

_, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(6, 8))
plt.imshow(bw, cmap='gray')
plt.title("Binary Map (White = Free, Black = Obstacle)")
plt.axis("off")
plt.show()

# STEP 3: Resize for Grid

grid_rows = 120
grid_cols = 60

bw = cv2.resize(bw, (grid_cols, grid_rows), interpolation=cv2.INTER_NEAREST)

plt.figure(figsize=(6, 8))
plt.imshow(bw, cmap='gray')
plt.xticks(np.arange(0, grid_cols + 1, 5))
plt.yticks(np.arange(0, grid_rows + 1, 10))
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.xlabel("Columns (X)")
plt.ylabel("Rows (Y)")
plt.title("Occupancy Grid Representation")
plt.tight_layout()
plt.show()


# Map size in Webots (metres)

map_width = 5.03
map_height = 9.725

cell_width = map_width / grid_cols
cell_height = map_height / grid_rows

# Robot start position from Webots
robot_x = 2.0
robot_y = 1.0

# Goal position in Webots world coordinates
goal_x = 2.0
goal_y = 3.0


# STEP 4: Convert to 0/1 Grid
# 0 = free
# 1 = obstacle

grid = np.zeros_like(bw, dtype=np.uint8)
grid[bw == 0] = 1
grid[bw == 255] = 0

print("Grid values:", np.unique(grid))


# STEP 5: Obstacle Inflation

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
grid_inflated = cv2.dilate(grid, kernel, iterations=1)

# Distance transform for wall penalty
# free space = 1, obstacle = 0
dist_transform = cv2.distanceTransform((1 - grid_inflated).astype(np.uint8), cv2.DIST_L2, 3)

# Display inflated map
inflated_display = np.ones_like(grid_inflated) * 255
inflated_display[grid_inflated == 1] = 0

plt.figure(figsize=(6, 8))
plt.imshow(inflated_display, cmap='gray')
plt.title("Inflated Obstacles (White = Free, Black = Obstacle)")
plt.axis("off")
plt.show()


# STEP 6: Convert Webots world -> grid

def world_to_grid(x, y, map_width, map_height, cell_width, cell_height, rows, cols):
    col = int((x + map_width / 2) / cell_width)
    row = int((map_height / 2 - y) / cell_height)

    col = max(0, min(cols - 1, col))
    row = max(0, min(rows - 1, row))

    return (row, col)

start = world_to_grid(robot_x, robot_y, map_width, map_height,
                      cell_width, cell_height, grid_rows, grid_cols)

goal = world_to_grid(goal_x, goal_y, map_width, map_height,
                     cell_width, cell_height, grid_rows, grid_cols)

print("Start grid:", start, "value:", grid_inflated[start])
print("Goal grid:", goal, "value:", grid_inflated[goal])

if grid_inflated[start] == 1:
    raise ValueError("Start inside obstacle")

if grid_inflated[goal] == 1:
    raise ValueError("Goal inside obstacle")


# STEP 7: Choose connectivity

connectivity = 8   # change to 4 or 8


# STEP 8: A* Algorithm

def astar(grid, start, goal, dist_transform, connectivity=4, use_wall_penalty=True):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}
    visited = set()

    def heuristic(a, b):
        if connectivity == 4:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        elif connectivity == 8:
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        else:
            raise ValueError("Connectivity must be 4 or 8")

    while open_set:
        _, current = heapq.heappop(open_set)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        if connectivity == 4:
            neighbors = [
                ((current[0] - 1, current[1]), 1.0),
                ((current[0] + 1, current[1]), 1.0),
                ((current[0], current[1] - 1), 1.0),
                ((current[0], current[1] + 1), 1.0)
            ]
        else:
            neighbors = [
                ((current[0] - 1, current[1]), 1.0),
                ((current[0] + 1, current[1]), 1.0),
                ((current[0], current[1] - 1), 1.0),
                ((current[0], current[1] + 1), 1.0),
                ((current[0] - 1, current[1] - 1), math.sqrt(2)),
                ((current[0] - 1, current[1] + 1), math.sqrt(2)),
                ((current[0] + 1, current[1] - 1), math.sqrt(2)),
                ((current[0] + 1, current[1] + 1), math.sqrt(2))
            ]

        for n, move_cost in neighbors:
            r, c = n

            if not (0 <= r < rows and 0 <= c < cols):
                continue

            if grid[r, c] == 1:
                continue

            # Prevent diagonal corner cutting for 8-neighbour
            dr = r - current[0]
            dc = c - current[1]
            if abs(dr) == 1 and abs(dc) == 1:
                if grid[current[0] + dr, current[1]] == 1 or grid[current[0], current[1] + dc] == 1:
                    continue

            wall_penalty = 0.0
            if use_wall_penalty:
                d = dist_transform[r, c]
                if d < 4:
                    wall_penalty = (4 - d) * 2

            tentative_g = g_score[current] + move_cost + wall_penalty

            if n not in g_score or tentative_g < g_score[n]:
                g_score[n] = tentative_g
                f_score = tentative_g + heuristic(n, goal)
                heapq.heappush(open_set, (f_score, n))
                came_from[n] = current

    return None


# STEP 9: Raw A* path (no wall penalty)

raw_path = astar(
    grid_inflated, start, goal,
    dist_transform,
    connectivity=connectivity,
    use_wall_penalty=False
)

if raw_path is None:
    raise ValueError("No raw A* path found!")

print("Raw A* path found:", raw_path)

# Keep waypoint reduction SAME as your working code
raw_reduced_path = raw_path[::3]
if raw_reduced_path[-1] != raw_path[-1]:
    raw_reduced_path.append(raw_path[-1])

print("Reduced Raw Path:", raw_reduced_path)

# STEP 10: Adapted A* path (with wall penalty)

adapted_path = astar(
    grid_inflated, start, goal,
    dist_transform,
    connectivity=connectivity,
    use_wall_penalty=True
)

if adapted_path is None:
    raise ValueError("No adapted A* path found!")

print("Adapted A* path found:", adapted_path)

# Keep waypoint reduction SAME as your working code
adapted_reduced_path = adapted_path[::3]
if adapted_reduced_path[-1] != adapted_path[-1]:
    adapted_reduced_path.append(adapted_path[-1])

print("Reduced Adapted Path:", adapted_reduced_path)


# STEP 11: Draw Raw A* Graph

path_display = np.ones_like(grid_inflated) * 255
path_display[grid_inflated == 1] = 0

plt.figure(figsize=(7, 8))
plt.imshow(path_display, cmap='gray')

if raw_reduced_path:
    path_np = np.array(raw_reduced_path)
    plt.plot(
        path_np[:, 1], path_np[:, 0],
        color='blue',
        linewidth=2.5,
        linestyle='-'
    )

start_plot = plt.scatter(
    start[1], start[0],
    c='green',
    s=150,
    marker='o'
)

goal_plot = plt.scatter(
    goal[1], goal[0],
    c='red',
    s=180,
    marker='*'
)

plt.xlabel("X (Grid Columns)", fontsize=12)
plt.ylabel("Y (Grid Rows)", fontsize=12)
plt.xticks(np.arange(0, grid_cols + 1, 10))
plt.yticks(np.arange(0, grid_rows + 1, 10))
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.title("A* Algorithm waypoints")

plt.legend(
    [start_plot, goal_plot],
    ["Start (circle)", "Goal (star)"],
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    borderaxespad=0.0
)

plt.tight_layout()
plt.show()


# STEP 12: Draw Adapted A* Graph

plt.figure(figsize=(7, 8))
plt.imshow(path_display, cmap='gray')

if adapted_reduced_path:
    path_np = np.array(adapted_reduced_path)
    plt.plot(
        path_np[:, 1], path_np[:, 0],
        color='blue',
        linewidth=2.5,
        linestyle='-'
    )

start_plot = plt.scatter(
    start[1], start[0],
    c='green',
    s=150,
    marker='o'
)

goal_plot = plt.scatter(
    goal[1], goal[0],
    c='red',
    s=180,
    marker='*'
)

plt.xlabel("X (Grid Columns)", fontsize=12)
plt.ylabel("Y (Grid Rows)", fontsize=12)
plt.xticks(np.arange(0, grid_cols + 1, 10))
plt.yticks(np.arange(0, grid_rows + 1, 10))
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.title("A* Algorithm Adapted for DWA Execution")

plt.legend(
    [start_plot, goal_plot],
    ["Start (circle)", "Goal (star)"],
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    borderaxespad=0.0
)

plt.tight_layout()
plt.show()


# STEP 13: Convert Adapted Path Grid -> World Coordinates

def grid_to_world(row, col, map_width, map_height, cell_width, cell_height):
    x = col * cell_width - map_width / 2 + cell_width / 2
    y = map_height / 2 - row * cell_height - cell_height / 2
    return (x, y)

waypoints = []

for r, c in adapted_reduced_path:
    wx, wy = grid_to_world(r, c, map_width, map_height, cell_width, cell_height)
    waypoints.append((wx, wy))

waypoints = np.array(waypoints)

# Save waypoints for Webots
save_path = r"C:\Users\Hari\Desktop\Mobile robotics\Contribution\controllers\ASTARandDWAcontroller\astar_waypoints.npy"
np.save(save_path, waypoints)

print("Waypoints:\n", waypoints)
print("The waypoints were saved successfully at:", save_path)


# STEP 14: Plot Adapted Path in World Coordinates

plt.figure(figsize=(8, 10))

plt.plot(
    waypoints[:, 0], waypoints[:, 1],
    '--o',
    linewidth=2.5,
    markersize=4,
    label='A* Waypoints for Webots'
)

plt.scatter(waypoints[0, 0], waypoints[0, 1], c='green', s=120, marker='s', label='Start')
plt.scatter(waypoints[-1, 0], waypoints[-1, 1], c='red', s=150, marker='*', label='Goal')

plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("A* Adapted Path for DWA in Webots World Coordinates")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.show()

