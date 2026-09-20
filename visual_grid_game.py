"""Lab 02: run with --agent simple, --agent model, or --compare.
Lab modifications are marked by step. Uses only the Python standard library.
"""
import argparse
import tkinter as tk

# LAB 02 - STEP 1.1: Heading is necessary for relative actions and wall_ahead.
# North, East, South, West; grid Y increases upward.
DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))


class VisualGridHuntGame:
    """Static Lab 02 comparison world, adapted from the Lab 01 environment.

    Opponents are disabled to isolate the effect of agent memory.
    Lab 01 traps and their penalties remain supported, but the comparison
    uses no traps so penalties cannot obscure the architecture demonstration.
    """

    def __init__(self, width=7, height=7, walls=(), food=((3, 3),),
                 traps=(), max_steps=300):
        self.width, self.height = width, height
        self.agent_pos = [0, 0]
        self.facing = 0
        self.walls = set(walls)
        self.food_positions = set(food)
        self.toxic_traps = set(traps)
        self.opponents = []
        self.score = 0
        self.steps = 0
        self.collision = False
        self.max_steps = max_steps
        self.bumped = False
        for cell in self.walls | self.food_positions | self.toxic_traps:
            if not (0 <= cell[0] < width and 0 <= cell[1] < height):
                raise ValueError("All cells must lie inside the grid")
        if (0, 0) in self.walls | self.toxic_traps:
            raise ValueError("The start must be safe")
        if self.food_positions & self.walls or self.toxic_traps & (self.walls | self.food_positions):
            raise ValueError("Walls, food and traps must not overlap")

    # ===== LAB 02 - STEP 1.1: Local sensors only =====
    def ahead(self):
        dx, dy = DIRECTIONS[self.facing]
        return self.agent_pos[0] + dx, self.agent_pos[1] + dy

    def blocked(self, cell):
        x, y = cell
        return not (0 <= x < self.width and 0 <= y < self.height) or cell in self.walls

    def get_percept(self):
        # No global position, heading, map, food count or opponent coordinates.
        # food_here checks the current cell; wall_ahead checks the next cell.
        # bumped is local feedback about the previous movement attempt.
        return {
            "wall_ahead": self.blocked(self.ahead()),
            "food_here": tuple(self.agent_pos) in self.food_positions,
            "bumped": self.bumped,
            "smells_toxin": tuple(self.agent_pos) in self.toxic_traps,
        }
    # ===== END STEP 1.1: Sensors =====

    # ===== LAB 02 - STEP 1.1: Supporting relative actuators =====
    def execute_action(self, action):
        if action not in {"turn_left", "turn_right", "move_forward", "suck", "stay"}:
            raise ValueError("Unknown action: " + str(action))
        self.steps += 1
        self.bumped = False
        if action == "turn_left":
            self.facing = (self.facing - 1) % 4
        elif action == "turn_right":
            self.facing = (self.facing + 1) % 4
        elif action == "move_forward":
            target = self.ahead()
            if self.blocked(target):
                self.bumped = True
                self.score -= 5
            else:
                self.agent_pos = list(target)
        elif action == "suck":
            # Food is now collected explicitly, not automatically on movement.
            cell = tuple(self.agent_pos)
            if cell in self.food_positions:
                self.food_positions.remove(cell)
                self.score += 20
        # Preserve Lab 01's once-per-action trap occupancy penalty.
        if tuple(self.agent_pos) in self.toxic_traps:
            self.score -= 15
    # ===== END STEP 1.1: Actuators =====

    def is_done(self):
        # A finite limit lets us demonstrate a persistent loop safely.
        return not self.food_positions or self.steps >= self.max_steps


# ===== LAB 02 - STEP 1.2: Simple reflex architecture =====
class SimpleReflexAgent:
    # No __init__, previous action, visited set or other history.
    def sense_and_act(self, percept):
        if percept["food_here"]:          # Condition: food at current cell.
            return "suck"                # Action: collect it.
        if percept["wall_ahead"]:         # Condition: forward movement blocked.
            return "turn_left"           # Action: turn 90 degrees left.
        return "move_forward"            # Otherwise advance.
# ===== END STEP 1.2 =====


# ===== LAB 02 - STEP 1.3: Model-based reflex architecture =====
class ModelBasedAgent:
    def __init__(self):
        # Relative frame: (0, 0) means "where I started", not a sensed location.
        # Heading 0 means the initial direction; turns update this estimate.
        self.position = (0, 0)
        self.heading = 0
        self.last_action = None
        self.last_percept = None
        self.visited_cells = set()
        self.visit_counts = {}
        self.observed_edges = {}  # (relative cell, heading) -> blocked boolean
        self.target_heading = None

    def neighbour(self, heading):
        dx, dy = DIRECTIONS[heading]
        return self.position[0] + dx, self.position[1] + dy

    def update_state(self, percept):
        # TRANSITION MODEL: predict the effect of the previous action.
        # The world is static; turns work reliably and movement is one cell.
        if self.last_action == "turn_left":
            self.heading = (self.heading - 1) % 4
        elif self.last_action == "turn_right":
            self.heading = (self.heading + 1) % 4
        elif self.last_action == "move_forward" and not percept["bumped"]:
            self.position = self.neighbour(self.heading)

        # SENSOR MODEL: interpret local feedback in the estimated state.
        # bumped=True prevents falsely recording a successful movement.
        # wall_ahead describes the edge in front of this relative pose.
        self.observed_edges[(self.position, self.heading)] = percept["wall_ahead"]
        self.last_percept = dict(percept)
        self.visited_cells.add(self.position)
        if self.last_action is None or (self.last_action == "move_forward" and not percept["bumped"]):
            self.visit_counts[self.position] = self.visit_counts.get(self.position, 0) + 1
        if self.last_action == "move_forward":
            self.target_heading = None

    def remember_action(self, action):
        self.last_action = action
        return action

    def sense_and_act(self, percept):
        # Update memory BEFORE matching a condition-action rule.
        self.update_state(percept)
        if percept["food_here"]:
            return self.remember_action("suck")

        # IF some directions are unobserved, THEN turn to sense them.
        # Four headings are inspected sequentially with the same front sensor.
        if any((self.position, h) not in self.observed_edges for h in range(4)):
            return self.remember_action("turn_right")

        # IF a route has not been selected, THEN prefer the least visited
        # open neighbour. Unvisited neighbours have count zero.
        # This is a memory-conditioned rule, not a search through future paths.
        if self.target_heading is None:
            open_headings = [h for h in range(4)
                             if not self.observed_edges[(self.position, h)]]
            if not open_headings:
                return self.remember_action("stay")
            self.target_heading = min(
                open_headings,
                key=lambda h: (self.visit_counts.get(self.neighbour(h), 0),
                               (h - self.heading) % 4))

        # IF aligned with the selected route, THEN move; otherwise turn.
        if self.heading == self.target_heading:
            if percept["wall_ahead"]:
                self.target_heading = None
                return self.remember_action("turn_right")
            return self.remember_action("move_forward")
        if (self.target_heading - self.heading) % 4 == 3:
            return self.remember_action("turn_left")
        return self.remember_action("turn_right")
# ===== END STEP 1.3 =====


def make_demo():
    # A U-shaped wall is inside the room. Reflex follows the outer boundary
    # forever and misses the food inside the U. Both agents start identically.
    walls = {(2, 2), (3, 2), (4, 2), (2, 3), (4, 3), (2, 4), (4, 4)}
    return VisualGridHuntGame(walls=walls)


def compare():
    # Observer-only diagnostics: true coordinates NEVER enter either agent.
    for agent_type in (SimpleReflexAgent, ModelBasedAgent):
        env, agent = make_demo(), agent_type()
        seen = {}
        repeat = None
        while not env.is_done():
            signature = (tuple(env.agent_pos), env.facing, frozenset(env.food_positions))
            if signature in seen and repeat is None:
                repeat = (seen[signature], env.steps)
            seen.setdefault(signature, env.steps)
            env.execute_action(agent.sense_and_act(env.get_percept()))
        print(f"{agent_type.__name__}: steps={env.steps}, score={env.score}, "
              f"food_remaining={len(env.food_positions)}, first_repeated_world_state={repeat}")


class GridGameGUI:
    """Tkinter wrapper that dynamically scales cell sizes to keep larger grids on screen."""

    def __init__(self, root, agent_kind="simple"):
        self.root = root
        self.root.title("SE3062 - Lab 02")

        # LAB 02 - STEP 1.2 / 1.3: Both agents receive the same demonstration.
        self.env = make_demo()
        self.agent = SimpleReflexAgent() if agent_kind == "simple" else ModelBasedAgent()
        self.root.title("Lab 02 - " + type(self.agent).__name__)

        # Dynamically calculate cell size so the total canvas fits nicely within a 600x600 window ceiling
        max_canvas_dim = 600
        self.cell_size = max(20, min(max_canvas_dim // self.env.width, max_canvas_dim // self.env.height))

        canvas_w = self.env.width * self.cell_size
        canvas_h = self.env.height * self.cell_size

        self.canvas = tk.Canvas(root, width=canvas_w, height=canvas_h, bg="white")
        self.canvas.pack()

        self.label = tk.Label(root, text="Score: 0 | Steps: 0", font=("Arial", 14))
        self.label.pack(pady=10)

        self.btn = tk.Button(root, text="Start Simulation", command=self.run_loop, font=("Arial", 12), bg="#000066",
                             fg="white")
        self.btn.pack(pady=5)

        self.draw_grid()

    def draw_grid(self):
        self.canvas.delete("all")

        for x in range(self.env.width):
            for y in range(self.env.height):
                x1 = x * self.cell_size
                y1 = (self.env.height - 1 - y) * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                color = "#f1f5f9" if (x, y) not in self.env.walls else "#64748b"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="#cbd5e1")

                # Only draw text if cell is large enough
                if self.cell_size >= 40 and (x, y) in self.env.walls:
                    self.canvas.create_text(x1 + self.cell_size / 2, y1 + self.cell_size / 2, text="W", fill="white",
                                            font=("Arial", 8, "bold"))

        # ===== LAB 01 - STEP 2.3 (B): Draw Toxic Traps =====
        # Render each trap as a purple diamond, as required by the lab.
        # Draw traps before characters so characters remain visible on top.
        # These GUI shapes are visible to the human; they do not give the agent
        # the full trap map through get_percept().
        for tx, ty in self.env.toxic_traps:
            # Convert the trap's grid location to the centre of its canvas cell.
            # Invert Y because canvas coordinates increase downward.
            cx = (tx + 0.5) * self.cell_size
            cy = (self.env.height - ty - 0.5) * self.cell_size
            # Scale the diamond to fit inside its cell.
            radius = self.cell_size * 0.35
            # Connect the top, right, bottom and left vertices of the diamond.
            self.canvas.create_polygon(
                cx, cy - radius, cx + radius, cy,
                cx, cy + radius, cx - radius, cy,
                fill="#9333ea", outline="#581c87")
        # ===== END STEP 2.3 (B) =====

        for fx, fy in self.env.food_positions:
            offset = self.cell_size * 0.25
            x1 = fx * self.cell_size + offset
            y1 = (self.env.height - 1 - fy) * self.cell_size + offset
            self.canvas.create_oval(x1, y1, x1 + self.cell_size * 0.5, y1 + self.cell_size * 0.5, fill="#f59e0b",
                                    outline="#d97706")

        for ox, oy in self.env.opponents:
            offset = self.cell_size * 0.2
            x1 = ox * self.cell_size + offset
            y1 = (self.env.height - 1 - oy) * self.cell_size + offset
            self.canvas.create_rectangle(x1, y1, x1 + self.cell_size * 0.6, y1 + self.cell_size * 0.6, fill="#990000",
                                         outline="#7a0000")

        ax, ay = self.env.agent_pos
        offset = self.cell_size * 0.15
        x1 = ax * self.cell_size + offset
        y1 = (self.env.height - 1 - ay) * self.cell_size + offset
        self.canvas.create_oval(x1, y1, x1 + self.cell_size * 0.7, y1 + self.cell_size * 0.7, fill="#000066",
                                outline="#1e3a8a")

        # LAB 02 - STEP 1.1: Show heading to the human, not in the percept.
        dx, dy = DIRECTIONS[self.env.facing]
        cx = (ax + 0.5) * self.cell_size
        cy = (self.env.height - ay - 0.5) * self.cell_size
        self.canvas.create_line(cx, cy, cx + dx * self.cell_size * 0.3,
                                cy - dy * self.cell_size * 0.3,
                                fill="white", width=3, arrow=tk.LAST)

    def run_loop(self):
        self.btn.config(state="disabled")

        def step():
            if not self.env.is_done():
                # LAB 02 - STEP 1.2 / 1.3: Replace random moves with the agent loop.
                # Only the local percept is supplied; no environment reference.
                percept = self.env.get_percept()
                action = self.agent.sense_and_act(percept)
                self.env.execute_action(action)

                self.draw_grid()
                self.label.config(text=f"Score: {self.env.score} | Steps: {self.env.steps} | Action: {action}")
                self.root.after(250, step)
            else:
                end_text = f"Collision! Game Over! Final Score: {self.env.score}" if self.env.collision else f"Finished! Final Score: {self.env.score}"
                self.label.config(text=end_text)
                self.btn.config(state="disabled")

        step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=("simple", "model"), default="simple")
    parser.add_argument("--compare", action="store_true", help="Run both agents without a GUI")
    args = parser.parse_args()
    if args.compare:
        compare()
    else:
        root = tk.Tk()
        app = GridGameGUI(root, agent_kind=args.agent)
        root.mainloop()
