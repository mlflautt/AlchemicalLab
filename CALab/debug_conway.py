"""
Debug Conway's Game of Life patterns
"""

import numpy as np
import matplotlib.pyplot as plt

def test_glider():
    """Test if glider survives correctly."""
    print("Testing Glider Pattern...")
    
    # Create minimal grid with glider
    grid = np.zeros((10, 10), dtype=bool)
    
    # Standard glider
    glider = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ], dtype=bool)
    
    grid[2:5, 2:5] = glider
    
    print("Initial glider:")
    print_grid(grid)
    
    # Test CA rules
    for gen in range(5):
        neighbors = count_neighbors(grid)
        new_grid = np.zeros_like(grid)
        
        # Conway's rules: B3/S23
        # Birth: exactly 3 neighbors
        new_grid |= (~grid) & (neighbors == 3)
        # Survival: 2 or 3 neighbors
        new_grid |= grid & ((neighbors == 2) | (neighbors == 3))
        
        grid = new_grid
        print(f"\\nGeneration {gen + 1}:")
        print_grid(grid)
        print(f"Alive count: {np.sum(grid)}")

def count_neighbors(grid):
    """Count neighbors using convolution."""
    from scipy import ndimage
    kernel = np.array([[1, 1, 1],
                      [1, 0, 1],
                      [1, 1, 1]])
    
    padded = np.pad(grid, 1, mode='constant', constant_values=0)  # Use zeros padding instead of wrap
    neighbors = ndimage.convolve(padded, kernel, mode='constant')[1:-1, 1:-1]
    return neighbors

def print_grid(grid):
    """Print grid with symbols."""
    for row in grid:
        print(''.join('█' if cell else '░' for cell in row))

def test_oscillators():
    """Test oscillator patterns."""
    print("\\n" + "="*30)
    print("Testing Oscillator Patterns...")
    
    # Blinker
    grid = np.zeros((5, 5), dtype=bool)
    grid[2, 1:4] = True  # Horizontal blinker
    
    print("\\nBlinker test:")
    print("Generation 0:")
    print_grid(grid)
    
    for gen in range(3):
        neighbors = count_neighbors(grid)
        new_grid = np.zeros_like(grid)
        
        new_grid |= (~grid) & (neighbors == 3)
        new_grid |= grid & ((neighbors == 2) | (neighbors == 3))
        
        grid = new_grid
        print(f"\\nGeneration {gen + 1}:")
        print_grid(grid)

def create_stable_world():
    """Create a world with stable patterns that work."""
    size = (60, 60)
    grid = np.zeros(size, dtype=bool)
    
    # Add multiple gliders in different positions
    glider = np.array([
        [0, 1, 0],
        [0, 0, 1], 
        [1, 1, 1]
    ], dtype=bool)
    
    # Gliders going in different directions
    positions = [(5, 5), (15, 35), (35, 15), (45, 45)]
    
    for i, (y, x) in enumerate(positions):
        rotated_glider = np.rot90(glider, i)  # Rotate each glider
        h, w = rotated_glider.shape
        if y + h < size[0] and x + w < size[1]:
            grid[y:y+h, x:x+w] |= rotated_glider
    
    # Add block (stable)
    grid[25:27, 25:27] = True
    
    # Add beehive (stable)
    beehive = np.array([
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 0]
    ], dtype=bool)
    grid[10:13, 10:14] = beehive
    
    # Add pulsar (oscillator, period 3)
    pulsar_pattern = [
        "  ███   ███  ",
        "             ",
        "█    █ █    █",
        "█    █ █    █", 
        "█    █ █    █",
        "  ███   ███  ",
        "             ",
        "  ███   ███  ",
        "█    █ █    █",
        "█    █ █    █",
        "█    █ █    █",
        "             ",
        "  ███   ███  "
    ]
    
    for i, row in enumerate(pulsar_pattern):
        for j, char in enumerate(row):
            if char == '█' and 20+i < size[0] and 20+j < size[1]:
                grid[20+i, 20+j] = True
    
    return grid

def visualize_working_life():
    """Create and visualize a working Game of Life."""
    plt.style.use('dark_background')
    
    grid = create_stable_world()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Conway's Game of Life - Working Patterns", color='white', fontsize=16)
    
    # Show evolution over time
    generations = [0, 10, 20, 50]
    
    for idx, target_gen in enumerate(generations):
        current_grid = grid.copy()
        
        # Evolve to target generation
        for gen in range(target_gen):
            neighbors = count_neighbors(current_grid)
            new_grid = np.zeros_like(current_grid)
            
            new_grid |= (~current_grid) & (neighbors == 3)
            new_grid |= current_grid & ((neighbors == 2) | (neighbors == 3))
            
            current_grid = new_grid
        
        ax = axes[idx // 2, idx % 2]
        ax.imshow(current_grid, cmap='binary', interpolation='nearest')
        ax.set_title(f"Generation {target_gen}", color='white')
        ax.axis('off')
        
        print(f"Generation {target_gen}: {np.sum(current_grid)} alive cells")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Conway's Game of Life Pattern Debug")
    print("="*40)
    
    # Test individual patterns
    test_glider()
    test_oscillators()
    
    # Show working visualization
    print("\\nCreating working Game of Life visualization...")
    visualize_working_life()