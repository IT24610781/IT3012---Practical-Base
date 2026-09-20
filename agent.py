"""SE3062 Lab 03: uninformed graph search; changes labelled by lab step."""
import random
from collections import deque
import heapq
from itertools import count


class GreedyGridAgent:
    """Original random baseline, retained for compatibility."""
    def sense_and_act(self, percept):
        return random.choice(['Up', 'Down', 'Left', 'Right'])


class SearchAgent:
    # STEP 1.3: Store a complete plan and select the active algorithm.
    def __init__(self, active_algo='BFS'):
        self.plan = []
        self.active_algo = active_algo.upper()
        self.status = 'Ready'

    # STEP 1.2: Shared transition model; each legal move costs one unit.
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

    # STEP 1.2 - BFS: FIFO queue, shallowest states first.
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

    # STEP 1.2 - DFS: LIFO stack, most recently generated states first.
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

    # STEP 1.2 - UCS: priority queue ordered by total path cost g(n).
    def ucs_search(self, start, goal, grid_size, walls):
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
            for child, action in self.successors(state, grid_size, walls):
                new_cost = cost + 1
                if new_cost < reached.get(child, float('inf')):
                    reached[child] = new_cost
                    parent[child] = (state, action)
                    heapq.heappush(frontier, (new_cost, next(order), child))
        return None

    # STEP 1.3: Find the closest REACHABLE food by actual grid distance.
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

    # STEP 1.3: Plan offline, then execute one stored action per call.
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
            if self.active_algo not in searches:
                raise ValueError('Choose BFS, DFS or UCS')
            self.plan = searches[self.active_algo](start, goal, grid_size, walls)
            if self.plan is None:
                self.plan = []
                self.status = 'No reachable food'
                return 'Stay'
            self.status = f'{self.active_algo}: planning to {goal}'
        return self.plan.pop(0)
