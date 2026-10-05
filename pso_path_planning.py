"""
Swarm Intelligence Lab - Assignment 1
Swarm-Based Path Planning with Obstacles

Student: Hamza Javed
Enrollment / Seed: 064
GitHub: hamza219676
"""

import math
import random
from collections import deque

import numpy as np

ROLL_NUMBER = 64
GRID_SIZE = 20
OBSTACLE_RATIO = 0.18

def get_neighbors(cell):
    row, col = cell
    directions = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    for dr, dc in directions:
        nr, nc = row + dr, col + dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
            yield nr, nc

def path_exists(start, goal, obstacles):
    queue = deque([start])
    visited = {start}
    while queue:
        current = queue.popleft()
        if current == goal:
            return True
        for neighbor in get_neighbors(current):
            if neighbor not in obstacles and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False

def generate_problem(seed=ROLL_NUMBER):
    rng = random.Random(seed)
    cells = [(r,c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
    obstacle_count = int(GRID_SIZE * GRID_SIZE * OBSTACLE_RATIO)
    while True:
        obstacles = set(rng.sample(cells, obstacle_count))
        free_cells = [cell for cell in cells if cell not in obstacles]
        start, goal = rng.sample(free_cells, 2)
        far_enough = math.dist(start, goal) >= GRID_SIZE * 0.55
        if far_enough and path_exists(start, goal, obstacles):
            return obstacles, start, goal

NUM_WAYPOINTS = 8
NUM_PARTICLES = 100
NUM_ITERATIONS = 300
C1 = 1.7
C2 = 1.7
MAX_VELOCITY = 3.0
COLLISION_PENALTY = 250.0

def bresenham_line(point_a, point_b):
    r0, c0 = [int(round(x)) for x in point_a]
    r1, c1 = [int(round(x)) for x in point_b]
    points = []
    dr = abs(r1-r0); dc = abs(c1-c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    error = dr - dc
    while True:
        points.append((r0,c0))
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * error
        if e2 > -dc:
            error -= dc; r0 += sr
        if e2 < dr:
            error += dr; c0 += sc
    return points

def decode_particle(position, start, goal):
    points = [start]
    for i in range(NUM_WAYPOINTS):
        points.append((position[2*i], position[2*i+1]))
    points.append(goal)
    return points

def calculate_fitness(position, start, goal, obstacles):
    points = decode_particle(position, start, goal)
    total_length = 0.0
    collisions = 0
    turning_penalty = 0.0
    previous_direction = None
    for point_a, point_b in zip(points, points[1:]):
        total_length += math.dist(point_a, point_b)
        collisions += sum(cell in obstacles for cell in bresenham_line(point_a, point_b))
        direction = (point_b[0]-point_a[0], point_b[1]-point_a[1])
        if previous_direction is not None:
            n1 = math.hypot(*previous_direction); n2 = math.hypot(*direction)
            if n1 > 0 and n2 > 0:
                cosine = (previous_direction[0]*direction[0] + previous_direction[1]*direction[1])/(n1*n2)
                cosine = max(-1.0, min(1.0, cosine))
                turning_penalty += 1.0 - cosine
        previous_direction = direction
    return total_length + COLLISION_PENALTY * collisions + 0.25 * turning_penalty

def pso_path_planning(start, goal, obstacles, seed=ROLL_NUMBER):
    rng = np.random.default_rng(seed)
    dimensions = 2 * NUM_WAYPOINTS
    positions = []
    start_array = np.array(start, dtype=float)
    goal_array = np.array(goal, dtype=float)
    for particle_index in range(NUM_PARTICLES):
        if particle_index < NUM_PARTICLES // 2:
            candidate = []
            for waypoint_index in range(1, NUM_WAYPOINTS+1):
                t = waypoint_index/(NUM_WAYPOINTS+1)
                base = start_array*(1-t) + goal_array*t
                noise = rng.normal(0, 3.5, 2)
                candidate.extend(np.clip(base+noise, 0, GRID_SIZE-1))
            positions.append(candidate)
        else:
            positions.append(rng.uniform(0, GRID_SIZE-1, dimensions))
    positions = np.array(positions, dtype=float)
    velocities = rng.uniform(-1.5,1.5,(NUM_PARTICLES,dimensions))
    personal_best = positions.copy()
    personal_best_values = np.array([calculate_fitness(p,start,goal,obstacles) for p in personal_best])
    best_index = int(np.argmin(personal_best_values))
    global_best = personal_best[best_index].copy()
    global_best_value = float(personal_best_values[best_index])
    history = [global_best_value]
    for iteration in range(NUM_ITERATIONS):
        inertia = 0.9 - (0.5 * iteration / NUM_ITERATIONS)
        r1 = rng.random((NUM_PARTICLES,dimensions))
        r2 = rng.random((NUM_PARTICLES,dimensions))
        velocities = inertia*velocities + C1*r1*(personal_best-positions) + C2*r2*(global_best-positions)
        velocities = np.clip(velocities,-MAX_VELOCITY,MAX_VELOCITY)
        positions = np.clip(positions+velocities,0,GRID_SIZE-1)
        values = np.array([calculate_fitness(p,start,goal,obstacles) for p in positions])
        improved = values < personal_best_values
        personal_best[improved] = positions[improved]
        personal_best_values[improved] = values[improved]
        best_index = int(np.argmin(personal_best_values))
        if personal_best_values[best_index] < global_best_value:
            global_best_value = float(personal_best_values[best_index])
            global_best = personal_best[best_index].copy()
        history.append(global_best_value)
    return global_best, global_best_value, history

def particle_to_grid_path(position, start, goal):
    points = decode_particle(position,start,goal)
    path = []
    for point_a, point_b in zip(points, points[1:]):
        segment = bresenham_line(point_a, point_b)
        if path:
            segment = segment[1:]
        path.extend(segment)
    cleaned = []
    for cell in path:
        if not cleaned or cell != cleaned[-1]:
            cleaned.append(cell)
    return cleaned

def path_length(path):
    return sum(math.dist(a,b) for a,b in zip(path,path[1:]))

def save_visualization(obstacles,start,goal,path,output_file):
    import matplotlib.pyplot as plt
    grid = np.zeros((GRID_SIZE,GRID_SIZE))
    for row,col in obstacles:
        grid[row,col] = 1
    plt.figure(figsize=(8,8))
    plt.imshow(grid,cmap="Greys",origin="upper",vmin=0,vmax=1)
    rows=[cell[0] for cell in path]; cols=[cell[1] for cell in path]
    plt.plot(cols,rows,marker="o",markersize=3,linewidth=2,label="PSO Path")
    plt.scatter(start[1],start[0],s=120,marker="o",label="Start")
    plt.scatter(goal[1],goal[0],s=140,marker="*",label="Goal")
    plt.title(f"PSO Path Planning - Hamza Javed - Seed {ROLL_NUMBER:03d}")
    plt.xlabel("Column"); plt.ylabel("Row")
    plt.xticks(range(GRID_SIZE)); plt.yticks(range(GRID_SIZE))
    plt.grid(True,linewidth=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(output_file,dpi=180); plt.close()

def save_convergence(history,output_file):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5)); plt.plot(history)
    plt.title("PSO Convergence"); plt.xlabel("Iteration"); plt.ylabel("Best Fitness / Cost")
    plt.grid(True,linewidth=0.4); plt.tight_layout(); plt.savefig(output_file,dpi=180); plt.close()

def main():
    obstacles,start,goal = generate_problem()
    best_particle,best_cost,history = pso_path_planning(start,goal,obstacles)
    final_path = particle_to_grid_path(best_particle,start,goal)
    collision_count = sum(cell in obstacles for cell in final_path)
    final_length = path_length(final_path)
    print("="*58)
    print("SWARM-BASED PATH PLANNING USING PSO")
    print("Student       : Hamza Javed")
    print(f"Enrollment    : {ROLL_NUMBER:03d}")
    print(f"Seed          : {ROLL_NUMBER}")
    print(f"Grid Size     : {GRID_SIZE} x {GRID_SIZE}")
    print(f"Obstacles     : {len(obstacles)}")
    print(f"Start         : {start}")
    print(f"Goal          : {goal}")
    print(f"Best PSO Cost : {best_cost:.4f}")
    print(f"Path Length   : {final_length:.4f}")
    print(f"Path Cells    : {len(final_path)}")
    print(f"Collisions    : {collision_count}")
    print("Final Path    :", final_path)
    print("="*58)
    save_visualization(obstacles,start,goal,final_path,"results/final_path.png")
    save_convergence(history,"results/convergence.png")
    print("SUCCESS: The final PSO path is obstacle-free." if collision_count==0 else "WARNING: The final path still contains obstacle collisions.")

if __name__ == "__main__":
    main()
