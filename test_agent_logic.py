from agent import SearchAgent


agent = SearchAgent(active_algo='AStar')

tile_facts = {
    (1, 0): [
        'TargetVisible',
        'HasDust',
        'BloodseekerMissing'
    ],
    (0, 1): [
        'TargetVisible',
        'HasDust'
    ]
}

# The direct route crosses (1, 0).
# A safe alternative exists through the upper row.
path = agent.astar_search(
    start_pos=(0, 0),
    goal_pos=(2, 0),
    walls=set(),
    grid_size=(3, 2),
    tile_facts=tile_facts
)

assert path is not None, "A safe alternative should exist."

moves = {
    'Up': (0, 1),
    'Down': (0, -1),
    'Left': (-1, 0),
    'Right': (1, 0)
}

position = (0, 0)

for action in path:
    dx, dy = moves[action]
    position = (position[0] + dx, position[1] + dy)

    assert position != (1, 0), "The agent entered the unsafe tile!"

assert position == (2, 0), "The agent did not reach the goal."

print("Safe path:", path)
print("Passed: A* avoided the unsafe tile.")

# Without the upper row, the only route crosses the unsafe tile.
blocked_path = agent.astar_search(
    start_pos=(0, 0),
    goal_pos=(2, 0),
    walls=set(),
    grid_size=(3, 1),
    tile_facts=tile_facts
)

assert blocked_path is None
print("Passed: A* rejected the only unsafe route.")