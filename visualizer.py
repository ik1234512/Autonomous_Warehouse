import matplotlib.pyplot as plt
from typing import List, Tuple
import config

def visualize_paths(env, path1: List[Tuple[int, int]], path2: List[Tuple[int, int]], 
                    start: Tuple[int, int], goal: Tuple[int, int], stats: dict):
    """Visualize paths cleanly side-by-side"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for ax, path, title in [(ax1, path1, "Manhattan Heuristic"), (ax2, path2, "Euclidean Heuristic")]:
        grid_display = [[1 if env.grid[y][x] == 1 else 0 for x in range(env.width)] for y in range(env.height)]
        ax.imshow(grid_display, cmap=config.COLOR_MAP, origin='upper', alpha=0.7)
        
        if path:
            ax.plot([p[0] for p in path], [p[1] for p in path], config.PATH_COLOR, linewidth=2, marker='o', markersize=4)
            
        ax.plot(start[0], start[1], config.START_COLOR, markersize=10, label='Start')
        ax.plot(goal[0], goal[1], config.GOAL_COLOR, markersize=10, label='Goal')
        
        heuristic_name = title.split()[0]
        ax.set_title(f"{title}\nNodes Expanded: {stats.get(heuristic_name, 'N/A')}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
    plt.tight_layout()
    plt.suptitle("Forklift Navigation Lab: Heuristic Comparison Suite", y=1.02, fontsize=12)
    plt.show()
