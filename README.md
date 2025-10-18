# Snake-With-Q-Learning

[Русская версия](./README.ru.md)

An interactive Snake game with an optional evolutionary Q-learning trainer. The game renders with `pixels`/`winit` and includes a semi-transparent control panel overlay. You can play manually or watch a population of agents learn to play via Q-learning with evolutionary strategies.

## Features

- Classic Snake on a fixed grid (800x600 window, 20px cells) with wrap-around edges.
- Smooth pixel rendering with a checkerboard grid background and snake head “eyes”.
- On-screen control panel with current score, length, speed, evolution status, epoch charts, and quick action buttons.
- Q-learning agent with compact, vision-based state encoding (20-bit key) and three actions: turn left, go straight, turn right.
- Evolutionary trainer (default population 24) running multiple agents in parallel, with elitism, mutation, and adaptive restarts on stagnation. Unique leader protection prevents premature epoch resets.
- Auto-save and auto-load of the best (champion) agent to/from `snake_agent.json`.
- GPU-aware training budget: if a GPU adapter is available, a higher steps-per-tick budget is used; toggle via keyboard.

## Controls

- Movement: Arrow keys or WASD
- Pause/Resume: P
- Restart game: R (when dead or from overlay button)
- Toggle evolution: E
- Adjust speed:
  - Manual play: `+` / `-` change tick time
  - Evolution: `+` doubles and `-` halves steps/frame (up to 100,000)
- Save best agent: S
- Toggle panel visibility: H
- Toggle acceleration budget (CPU/GPU): G (only shown if a GPU is available)
- Ultra-fast training (skip rendering, raise budget): U
- Show only best agent during training: B
- If built with feature `gpu-nn`: toggle experimental NN mode with N
- Quit: Esc or close window
- Mouse: Click panel buttons (Pause/Resume, Speed+, Restart, Save, Hide/Show)

## Build and Run

Prerequisites:
- Rust toolchain (stable)
- Windows (tested), but should work on other platforms supported by `pixels`/`winit`.

Run (debug):

```powershell
cargo run
```

Run (optimized):

```powershell
cargo run --release
```

On start, the app tries to load `snake_agent.json`. If found, evolution auto-starts using the loaded agent as a seed.

## How the learning works

### State encoding (vision + context)
The agent observes an 8-cell neighborhood around the snake head in a direction-relative frame (3x3 area forward). Each cell is encoded with 2 bits:
- 00 = empty
- 01 = danger (wall/body)
- 10 = apple
- 11 = unused

This uses 16 bits. Additionally:
- 2 bits: relative direction to the apple (left/straight/right)
- 2 bits: Manhattan distance category to the apple (4 buckets)

Total: 20-bit state key (~1M states).

### Actions
Three discrete actions relative to the current direction:
- 0 = turn left
- 1 = go straight
- 2 = turn right

### Rewards
- +10.0 for eating an apple, increasing slightly with length (+0.1 per current length)
- Death penalty depends on cause: -30.0 for self-collision; -12.0 otherwise
- Small step penalty (-0.005)
- Shaping: +0.05 when moving closer and -0.03 when moving away; additional +0.02 when within 3 cells of the apple

### QAgent parameters
- epsilon-greedy with decay (`epsilon`, `min_epsilon`, `decay`)
- learning rate `alpha`, discount `gamma`
- `steps` and `episodes` counters recorded per agent

### Evolutionary trainer
- Population of agents (default 24), each playing in its own game instance in parallel
- Per-epoch step limit with a “leader protection” exception that lets a unique best agent continue beyond the limit
- At epoch end, reproduction with elitism + mutations; multiple staged restart strategies on long stagnation, seeding from the global champion
- Tracks a global champion (best ever), with auto-save on improvement
- Agents are color-coded for visualization

## Code structure

- `src/main.rs` — the main application with game logic, rendering, Q-learning agent, and evolutionary trainer.
- `src/gpu_nn.rs` — optional experimental NN scaffolding (behind the `gpu-nn` feature).
- `snake_agent.json` — saved champion agent (created at runtime when saving).

## Tips

- To start training from scratch, delete `snake_agent.json` or press E to toggle training and let evolution run.
- At very high training speeds, frames are skipped and drawing can be disabled to maximize throughput.
- Grid/cell sizes are constants near the top of `main.rs` and can be adjusted as needed.

## License

MIT License — see LICENSE file.
