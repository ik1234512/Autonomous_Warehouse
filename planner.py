import heapq
import time
from typing import List, Tuple, Dict, Set
from models import Node
from environment import SurgicalLabEnvironment

class ForkliftPlanner:
    """A* path planner comparing different heuristics"""
    def __init__(self, environment: SurgicalLabEnvironment):
        self.env = environment
        
    def a_star(self, start: Tuple[int, int], goal: Tuple[int, int], heuristic_type: str = 'manhattan') -> Tuple[List[Tuple[int, int]], int, float]:
        start_time = time.time()
        nodes_expanded = 0
        
        # ---------------------------------------------------------
        # SETUP: We have initialized the starting node for you!
        # ---------------------------------------------------------
        start_node = Node(start[0], start[1])
        start_node.g = 0
        
        # Choose the correct heuristic based on the function argument
        if heuristic_type == 'manhattan':
            start_node.h = self.env.manhattan_distance(start[0], start[1], goal[0], goal[1])
        elif heuristic_type == 'chebyshev':
            start_node.h = self.env.chebyshev_distance(start[0], start[1], goal[0], goal[1])
        elif heuristic_type == 'euclidean':
            start_node.h = self.env.euclidean_distance(start[0], start[1], goal[0], goal[1])
        else:
            start_node.h = self.env.euclidean_distance(start[0], start[1], goal[0], goal[1])
        
        # Data Structures you will need
        open_set = []
        heapq.heappush(open_set, start_node)
        
        closed_set: Set[Node] = set()
        open_dict: Dict[Tuple[int, int], Node] = {(start[0], start[1]): start_node}
        
        # ---------------------------------------------------------
        # TODO 1: THE CORE LOOP
        # Loop as long as there are nodes in the open_set
        # ---------------------------------------------------------
        
        while open_set:
            
            # 1. Pop the node with the lowest f-score from open_set using heapq
            current = heapq.heappop(open_set)
            # 2. (Optional but recommended) Check if this node is stale using open_dict
            if open_dict.get((current.x, current.y)) != current:
                continue
            del open_dict[current.x, current.y]

            # 3. Check if the current node is the goal! 
            if current.x == goal[0] and current.y == goal[1]:
                path = []
                while current:
                    path.append((current.x, current.y))
                    current = current.parent

            # If yes, reconstruct the path using the .parent attributes, reverse it, and return it.
                
                path = path[: : -1]
                return path, nodes_expanded, time.time() - start_time
            
            # 4. Add the current node to the closed_set and increment nodes_expanded
            closed_set.add((current.x, current.y))
            nodes_expanded += 1
            # ---------------------------------------------------------
            # TODO 2: EVALUATING NEIGHBORS
            # ---------------------------------------------------------
            # 5. Use self.env.get_neighbors(current.x, current.y) to get valid moves
            for nx, ny in self.env.get_neighbors(current.x, current.y):
            
            # For each neighbor:
                # a. Skip if it is already in the closed_set
                
                if (nx, ny) in closed_set:
                    continue
            # b. Calculate the tentative_g score (current g + 1)
                g = current.g + 1
                if heuristic_type == 'manhattan':
                    h = self.env.manhattan_distance(nx, ny, goal[0], goal[1])
                elif heuristic_type == 'chebyshev':
                    h = self.env.chebyshev_distance(nx, ny, goal[0], goal[1])
                elif heuristic_type == 'euclidean':
                    h = self.env.euclidean_distance(nx, ny, goal[0], goal[1])
                else:
                    h = self.env.euclidean_distance(nx, ny, goal[0], goal[1])
            
            # c. If it's a new node OR the tentative_g is better than its existing g:
                 # - Update its parent, g, h, and f scores
                 # - Push it to the open_set
                 # - Update the open_dict

                neighbor = Node(nx, ny)
                neighbor.g = g
                neighbor.h = h
                neighbor.f = g + h
                neighbor.parent = current

                heapq.heappush(open_set, neighbor)
                open_dict[(nx, ny)] = neighbor
                
	
                    
        # Returns empty list if no path is found
        return ([], nodes_expanded, time.time() - start_time)
