import numpy as np
import random
import matplotlib.pyplot as plt

# Configuration & Seed setup using roll number
SEED = 136232
random.seed(SEED)
np.random.seed(SEED)

GRID_SIZE = 20
OBSTACLE_DENSITY = 0.25  # 25% of cells will be obstacles

def generate_grid(grid_size, density):
    grid = np.zeros((grid_size, grid_size), dtype=int)
    
    # Generate random obstacles
    for i in range(grid_size):
        for j in range(grid_size):
            if random.random() < density:
                grid[i, j] = 1  # 1 represents an obstacle
                
    # Generate start and goal on free cells (value 0)
    free_cells = [(r, c) for r in range(grid_size) for c in range(grid_size) if grid[r, c] == 0]
    start, goal = random.sample(free_cells, 2)
    
    return grid, start, goal

if __name__ == "__main__":
    grid, start, goal = generate_grid(GRID_SIZE, OBSTACLE_DENSITY)
    print(f"Grid Size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Start Position: {start}")
    print(f"Goal Position: {goal}")
    print(f"Total Obstacles Generated: {np.sum(grid)}")
    