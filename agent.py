"""SE3062 Lab 04: informed and uninformed graph search; changes labelled by lab step."""
import random
import math
from collections import deque
import heapq
from itertools import count


class GreedyGridAgent:
    """Original random baseline, retained for compatibility."""
    def sense_and_act(self, percept):
        return random.choice(['Up', 'Down', 'Left', 'Right'])


class SearchAgent:
    # LAB 03 - STEP 1.3: Store a complete plan and select the active algorithm.
    def __init__(self, active_algo='AStar', heuristic_type='manhattan'):
        self.plan = []
        # LAB 04 - STEP 1.3: Normalize AStar without breaking BFS/DFS/UCS.
        self.active_algo = 'AStar' if active_algo.upper() in ('ASTAR', 'A*') else active_algo.upper()
        if heuristic_type not in ('manhattan', 'euclidean'):
            raise ValueError('Choose manhattan or euclidean')
        self.heuristic_type = heuristic_type
        self.expanded_nodes = 0
        self.status = 'Ready'

    # LAB 03 - STEP 1.2: Shared transition model; each legal move costs one unit.
    @staticmethod
    def successors(state, grid_size, walls):
        x, y = state
        width, height = grid_size
        for action, dx, dy in [('Up', 0, 1), ('Right', 1, 0),
                               ('Down', 0, -1), ('Left', -1, 0)]:
            next_state = (x + dx, y + dy)
            if (0 <= next_state[0] < width and 0 <= next_state[1] < height
                    and next_state not in walls):
                yield next_state, action

    @staticmethod
    def reconstruct(parent, goal):
        # Store one parent per state, not a full path in every frontier entry.
        path = []
        while parent[goal] is not None:
            goal, action = parent[goal]
            path.append(action)
        return path[::-1]

    # LAB 03 - STEP 1.2 - BFS: FIFO queue, shallowest states first.
    def bfs_search(self, start, goal, grid_size, walls):
        walls = set(walls)
        frontier = deque([start])
        reached = {start}
        parent = {start: None}
        while frontier:
            state = frontier.popleft()
            if state == goal:
                return self.reconstruct(parent, state)
            for child, action in self.successors(state, grid_size, walls):
                if child not in reached:
                    reached.add(child)  # Mark on insertion to avoid duplicates.
                    parent[child] = (state, action)
                    frontier.append(child)
        return None  # No route exists; [] instead means start == goal.

    # LAB 03 - STEP 1.2 - DFS: LIFO stack, most recently generated states first.
    def dfs_search(self, start, goal, grid_size, walls):
        walls = set(walls)
        frontier = [start]
        reached = {start}
        parent = {start: None}
        while frontier:
            state = frontier.pop()
            if state == goal:
                return self.reconstruct(parent, state)
            for child, action in self.successors(state, grid_size, walls):
                if child not in reached:
                    reached.add(child)
                    parent[child] = (state, action)
                    frontier.append(child)
        return None

    # LAB 03 - STEP 1.2 - UCS: priority queue ordered by total path cost g(n).
    def ucs_search(self, start, goal, grid_size, walls):
        self.expanded_nodes = 0
        walls = set(walls)
        order = count()  # Stable FIFO tie-breaking for equal-cost entries.
        frontier = [(0, next(order), start)]
        reached = {start: 0}  # Best cost, not just a boolean visited flag.
        parent = {start: None}
        while frontier:
            cost, _, state = heapq.heappop(frontier)
            if cost != reached[state]:
                continue  # Ignore a stale, more expensive entry.
            if state == goal:  # Goal test when popped, not when generated.
                return self.reconstruct(parent, state)
            self.expanded_nodes += 1
            for child, action in self.successors(state, grid_size, walls):
                new_cost = cost + 1
                if new_cost < reached.get(child, float('inf')):
                    reached[child] = new_cost
                    parent[child] = (state, action)
                    heapq.heappush(frontier, (new_cost, next(order), child))
        return None

    # ===== LAB 04 - STEP 1.1: Heuristic functions =====
    def manhattan_distance(self, pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def euclidean_distance(self, pos, goal):
        return math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)
    # ===== END STEP 1.1 =====

    # ===== LAB 04 - STEP 1.2: A* graph search =====
    def astar_search(self, start_pos, goal_pos, walls, grid_size,
                     heuristic_type='manhattan'):
        heuristics = {'manhattan': self.manhattan_distance,
                      'euclidean': self.euclidean_distance}
        if heuristic_type not in heuristics:
            raise ValueError('Choose manhattan or euclidean')
        heuristic = heuristics[heuristic_type]
        start_pos, goal_pos = tuple(start_pos), tuple(goal_pos)
        walls = {tuple(w) for w in walls}
        self.expanded_nodes = 0
        for x, y in (start_pos, goal_pos):
            if not (0 <= x < grid_size[0] and 0 <= y < grid_size[1]) or (x, y) in walls:
                return None

        frontier = []
        reached_states = set()  # States closed AFTER popping from the queue.
        best_g = {start_pos: 0}  # Avoid equal/worse duplicate frontier entries.
        # Required tuple: (f_cost, g_cost, current_pos, path_taken).
        heapq.heappush(frontier, (heuristic(start_pos, goal_pos), 0, start_pos, []))
        while frontier:
            f_cost, g_cost, current_pos, path_taken = heapq.heappop(frontier)
            if current_pos in reached_states or g_cost != best_g[current_pos]:
                continue
            if current_pos == goal_pos:
                return path_taken  # Goal test on pop, not on generation.
            reached_states.add(current_pos)
            self.expanded_nodes += 1  # Count successor expansions, excluding goal.
            for neighbour, action in self.successors(current_pos, grid_size, walls):
                if neighbour in reached_states:
                    continue
                g_new = g_cost + 1  # Four-way grid: each move costs 1.
                if g_new < best_g.get(neighbour, float('inf')):
                    best_g[neighbour] = g_new
                    h_new = heuristic(neighbour, goal_pos)
                    f_new = g_new + h_new
                    heapq.heappush(frontier, (f_new, g_new, neighbour,
                                               path_taken + [action]))
        return None
        # Closing states permanently is valid here: both supported heuristics
        # are consistent on the static four-way unit-cost grid. Arbitrary
        # inconsistent heuristics may require reopening improved states.
    # ===== END STEP 1.2 =====

    # LAB 03 - STEP 1.3: Find the closest REACHABLE food by actual grid distance.
    # A separate BFS target-selection pass accounts for walls. Manhattan
    # distance alone can choose a nearby-looking but unreachable pellet.
    def closest_food(self, start, foods, grid_size, walls):
        frontier = deque([start])
        reached = {start}
        while frontier:
            state = frontier.popleft()
            if state in foods:
                return state
            for child, _ in self.successors(state, grid_size, walls):
                if child not in reached:
                    reached.add(child)
                    frontier.append(child)
        return None

    # LAB 03 - STEP 1.3: Plan offline, then execute one stored action per call.
    def sense_and_act(self, percept):
        if not self.plan:
            start = tuple(percept['agent_pos'])
            foods = {tuple(p) for p in percept['all_food']}
            walls = {tuple(p) for p in percept['walls']}
            grid_size = tuple(percept['grid_size'])
            if not foods:
                self.status = 'All food collected'
                return 'Stay'
            if start in foods:
                return 'Stay'  # Base game collects food at the resulting cell.
            goal = self.closest_food(start, foods, grid_size, walls)
            if goal is None:
                self.status = 'No reachable food'
                return 'Stay'
            searches = {'BFS': self.bfs_search, 'DFS': self.dfs_search,
                        'UCS': self.ucs_search}
            # LAB 04 - STEP 1.3: Integrate A* alongside previous algorithms.
            # all_food contains coordinates; remaining_food is only a count.
            if self.active_algo in searches:
                self.plan = searches[self.active_algo](start, goal, grid_size, walls)
            elif self.active_algo == 'AStar':
                self.plan = self.astar_search(start, goal, walls, grid_size,
                                               self.heuristic_type)
            else:
                raise ValueError('Choose BFS, DFS, UCS or AStar')
            if self.plan is None:
                self.plan = []
                self.status = 'No reachable food'
                return 'Stay'
            self.status = f'{self.active_algo}: planning to {goal}'
        return self.plan.pop(0)
