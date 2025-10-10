# 🗺️ A* Path Planning on a Binary Occupancy Grid

This project implements the **A\*** (A-star) path planning algorithm on a **binary occupancy grid**, representing a simplified 2D environment where free spaces and obstacles are encoded as `0` and `1` respectively. The objective is to compute the **shortest collision-free path** between a defined start and goal location.

---

## 📘 Project Overview

Path planning is a fundamental challenge in **robotics** and **autonomous navigation**. This notebook demonstrates the A* algorithm applied to a binary occupancy map generated from an image. It efficiently finds the shortest path using a heuristic-guided search strategy.

**Workflow Summary:**
1. Load an image and convert it into a binary occupancy grid.
2. Represent the environment as a graph of traversable and blocked cells.
3. Apply the **A\*** algorithm to find the optimal path between start and goal points.
4. Visualize the resulting path over the occupancy grid.

---

## ⚙️ Technical Details

### 1. Binary Occupancy Grid
The map image (`occupancy_map.png`) is converted into a binary grid where:
- `0` → Free cell (traversable)
- `1` → Obstacle (non-traversable)

python
```occupancy_grid = (np.asarray(occupancy_map_img) > 0).astype(int)```
Each cell in this grid acts as a discrete node in the search space.

2. A* Algorithm Overview
The A* algorithm computes the shortest path using the equation:
f(n) = g(n) + h(n)
Where:
g(n) = Cost from the start node to the current node
h(n) = Heuristic estimate (e.g., Euclidean or Manhattan distance) from the current node to the goal
f(n) = Total estimated cost through node n
Nodes are expanded in order of increasing f(n) values until the goal is reached.

3. Path Reconstruction

Once the goal is reached, the path is reconstructed using a predecessor dictionary:
```
def RecoverPath(s, g, pred):
    optimal_path = []
    current_vertex = g
    while current_vertex != s:
        optimal_path.append(current_vertex)
        current_vertex = pred[current_vertex]
    optimal_path.append(s)
    optimal_path.reverse()
    return optimal_path
```
🧩 Dependencies
Install the required Python libraries using:
pip install numpy matplotlib pillow

🚀 How to Run
Clone this repository:
```git clone https://github.com/<your-username>/Astar-Path-Planning.git
cd Astar-Path-Planning
```
Place your occupancy map image (occupancy_map.png) in the project directory.
Open the notebook and execute all cells:
jupyter notebook "Astar algorithm implementation.ipynb"
Observe the visualized optimal path on the occupancy grid.


🧠 Key Concepts Demonstrated

-Binary occupancy grid representation
-Heuristic-based graph search (A*)
-Path cost optimization and reconstruction
-Visualization of planning results in Python
-Application of search algorithms to robotics problems

🔮 Future Improvements
-Add diagonal movement support
-Integrate user-defined start and goal points
-Visualize search expansion (open/closed sets)
-Compare performance with Dijkstra’s or Greedy Best-First Search
-Extend to ROS-based occupancy grid maps

🧑‍💻 Author
Poojit Maddineni
Passionate about robotics, AI-driven navigation, and autonomous systems.
Feel free to connect and explore more projects! 🚀
