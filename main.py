import numpy as np
import random
import matplotlib.pyplot as plt

# Configuration & Seed setup using roll number
SEED = 136232
random.seed(SEED)
np.random.seed(SEED)

GRID_SIZE = 20
OBSTACLE_DENSITY = 0.25

def generate_grid(grid_size, density):
    grid = np.zeros((grid_size, grid_size), dtype=int)
    for i in range(grid_size):
        for j in range(grid_size):
            if random.random() < density:
                grid[i, j] = 1
                
    free_cells = [(r, c) for r in range(grid_size) for c in range(grid_size) if grid[r, c] == 0]
    start, goal = random.sample(free_cells, 2)
    return grid, start, goal

# --- PSO Implementation ---
NUM_PARTICLES = 30
MAX_ITER = 100
NUM_WAYPOINTS = 5

class Particle:
    def __init__(self, start, goal):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.position = np.random.uniform(0, GRID_SIZE, (NUM_WAYPOINTS, 2))
        self.velocity = np.random.uniform(-1, 1, (NUM_WAYPOINTS, 2))
        self.best_position = np.copy(self.position)
        self.best_score = float('inf')

def fitness(particle, grid, start, goal):
    full_path = np.vstack([start, particle.position, goal])
    # Fixed: np.linalg.norm instead of np.linalg_norm
    dist_cost = np.sum(np.linalg.norm(full_path[1:] - full_path[:-1], axis=1))
    
    penalty = 0
    for point in particle.position:
        r, c = int(np.clip(point[0], 0, GRID_SIZE - 1)), int(np.clip(point[1], 0, GRID_SIZE - 1))
        if grid[r, c] == 1:
            penalty += 500
            
    return dist_cost + penalty

def run_pso(grid, start, goal):
    particles = [Particle(start, goal) for _ in range(NUM_PARTICLES)]
    gbest_position = None
    gbest_score = float('inf')
    
    w, c1, c2 = 0.5, 1.5, 1.5

    for iteration in range(MAX_ITER):
        for p in particles:
            score = fitness(p, grid, start, goal)
            if score < p.best_score:
                p.best_score = score
                p.best_position = np.copy(p.position)
            if score < gbest_score:
                gbest_score = score
                gbest_position = np.copy(p.position)

        for p in particles:
            r1, r2 = np.random.rand(), np.random.rand()
            cognitive = c1 * r1 * (p.best_position - p.position)
            social = c2 * r2 * (gbest_position - p.position)
            p.velocity = w * p.velocity + cognitive + social
            p.position += p.velocity
            p.position = np.clip(p.position, 0, GRID_SIZE - 1)

    full_best_path = np.vstack([start, gbest_position, goal])
    return full_best_path, gbest_score

def plot_grid_and_path(grid, start, goal, path):
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap='binary', origin='upper')
    
    # Plot Start and Goal positions
    plt.plot(start[1], start[0], 'go', markersize=12, label='Start')
    plt.plot(goal[1], goal[0], 'ro', markersize=12, label='Goal')
    
    # Plot Planned Path
    plt.plot(path[:, 1], path[:, 0], 'b-o', linewidth=2, label='PSO Path')
    
    plt.title('Swarm-Based Path Planning (PSO)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
    plt.show()

if __name__ == "__main__":
    grid, start, goal = generate_grid(GRID_SIZE, OBSTACLE_DENSITY)
    best_path, best_score = run_pso(grid, start, goal)
    
    print(f"Grid Size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Start: {start} | Goal: {goal}")
    print(f"Optimal Path Fitness Score: {best_score:.2f}")
    
    plot_grid_and_path(grid, start, goal, best_path)
    