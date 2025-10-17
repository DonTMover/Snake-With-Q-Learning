//! Snake game with an optional evolutionary Q-learning trainer.
//!
//! This binary renders the classic Snake game using `pixels`/`winit` and includes
//! an on-screen control panel. When Evolution (E) is enabled, a population of
//! Q-learning agents is trained in parallel. The best agent (champion) is saved
//! to `snake_agent.json` and auto-loaded on the next run.
//!
//! Key controls:
//! - Arrows/WASD: move
//! - P: pause/resume
//! - R: restart
//! - E: toggle evolutionary training
//! - S: save best agent
//! - +/-: adjust speed (manual vs. evolution modes differ)
//! - H: show/hide control panel
//! - Esc: quit
//!
//! Learning summary:
//! - State: compact 20-bit encoding (vision of 8 cells around head + apple direction + distance bucket)
//! - Actions: turn left, go straight, turn right
//! - Rewards: +apple, -death, small step cost, shaping for distance improvement
//! - Evolution: elitism, mutation, and staged restarts on stagnation

#[cfg(feature = "gpu-nn-experimental")]
mod gpu_nn;

use ahash::AHashMap;
use pixels::{Error, Pixels, SurfaceTexture};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use wgpu::{Backends, Instance, PowerPreference};
use winit::dpi::LogicalSize;
use winit::event::{Event, VirtualKeyCode};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;
use winit_input_helper::WinitInputHelper;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;
const GRID_SIZE: u32 = 20;
const GRID_WIDTH: u32 = WIDTH / GRID_SIZE;
const GRID_HEIGHT: u32 = HEIGHT / GRID_SIZE;

/// Integer grid position (cell coordinates).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Pos {
    x: i32,
    y: i32,
}

impl Pos {
    /// Construct a new grid position.
    fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

/// Snake movement direction.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Dir {
    Up,
    Down,
    Left,
    Right,
}

/// Cause of death for reward shaping.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum DeathCause {
    None,
    SelfCollision,
}

/// Game state: snake body, apple, direction, score and flags.
struct Game {
    snake: VecDeque<Pos>,
    snake_set: HashSet<Pos>,
    dir: Dir,
    apple: Pos,
    alive: bool,
    score: usize,
    paused: bool,
    last_death: DeathCause,
}

impl Game {
    /// Create a new game with a short snake centered on the grid and a random apple.
    fn new() -> Self {
        let start_x = (GRID_WIDTH / 2) as i32;
        let start_y = (GRID_HEIGHT / 2) as i32;
        let mut snake = VecDeque::new();
        let mut snake_set = HashSet::new();
        let p0 = Pos::new(start_x, start_y);
        let p1 = Pos::new(start_x - 1, start_y);
        let p2 = Pos::new(start_x - 2, start_y);
        snake.push_back(p0);
        snake_set.insert(p0);
        snake.push_back(p1);
        snake_set.insert(p1);
        snake.push_back(p2);
        snake_set.insert(p2);

        let mut game = Self {
            snake,
            dir: Dir::Right,
            apple: Pos::new(0, 0),
            alive: true,
            score: 0,
            paused: false,
            snake_set,
            last_death: DeathCause::None,
        };
        game.place_apple();
        game
    }

    /// Place an apple on a random empty cell (not colliding with the snake).
    fn place_apple(&mut self) {
        let mut rng = rand::thread_rng();
        loop {
            let x = rng.gen_range(0..GRID_WIDTH as i32);
            let y = rng.gen_range(0..GRID_HEIGHT as i32);
            let p = Pos::new(x, y);
            if !self.snake_set.contains(&p) {
                self.apple = p;
                break;
            }
        }
    }

    /// Advance the game by one tick: move the snake, handle apple/self/wall collisions.
    fn update(&mut self) {
        if !self.alive || self.paused {
            return;
        }

        // reset death cause at the start of a tick
        self.last_death = DeathCause::None;

        let head = self.snake.front().unwrap();
        // Move head with toroidal wrapping across edges
        let mut new_x = head.x;
        let mut new_y = head.y;
        match self.dir {
            Dir::Up => new_y -= 1,
            Dir::Down => new_y += 1,
            Dir::Left => new_x -= 1,
            Dir::Right => new_x += 1,
        }
        if new_x < 0 {
            new_x = GRID_WIDTH as i32 - 1;
        } else if new_x >= GRID_WIDTH as i32 {
            new_x = 0;
        }
        if new_y < 0 {
            new_y = GRID_HEIGHT as i32 - 1;
        } else if new_y >= GRID_HEIGHT as i32 {
            new_y = 0;
        }
        let new_head = Pos::new(new_x, new_y);

        // Check collision with self (tail collision disallowed like before)
        if self.snake_set.contains(&new_head) {
            self.last_death = DeathCause::SelfCollision;
            self.alive = false;
            return;
        }

        self.snake.push_front(new_head);
        self.snake_set.insert(new_head);

        // Check if apple eaten
        if new_head == self.apple {
            self.score += 1;
            self.place_apple();
        } else if let Some(tail) = self.snake.pop_back() {
            self.snake_set.remove(&tail);
        }
    }

    /// Change movement direction, disallowing 180-degree turns.
    fn change_dir(&mut self, new_dir: Dir) {
        // Prevent 180 degree turns
        let opposite = match self.dir {
            Dir::Up => Dir::Down,
            Dir::Down => Dir::Up,
            Dir::Left => Dir::Right,
            Dir::Right => Dir::Left,
        };
        if new_dir != opposite {
            self.dir = new_dir;
        }
    }

    /// Draw the current game state to the frame buffer (RGBA8).
    fn draw(&self, frame: &mut [u8]) {
        // Clear screen with dark background
        clear_rgba(frame, 30, 30, 40, 255);

        // Draw grid
        for y in 0..GRID_HEIGHT {
            for x in 0..GRID_WIDTH {
                if (x + y) % 2 == 0 {
                    self.draw_rect(frame, x, y, 35, 35, 50);
                }
            }
        }

        // Draw apple (red)
        fill_cell_rgb(frame, self.apple.x as u32, self.apple.y as u32, 220, 50, 50);

        // Draw snake
        for (i, &pos) in self.snake.iter().enumerate() {
            if i == 0 {
                // Head (bright green)
                fill_cell_rgb(frame, pos.x as u32, pos.y as u32, 100, 255, 100);
                // Draw eyes based on direction
                self.draw_eyes(frame, &pos);
            } else {
                // Body (gradient green)
                let brightness = 200 - (i * 10).min(100) as u8;
                fill_cell_rgb(frame, pos.x as u32, pos.y as u32, 50, brightness, 50);
            }
        }

        // Draw score
        if !self.alive {
            // Game over overlay
            draw_text(
                frame,
                "GAME OVER",
                WIDTH / 2 - 80,
                HEIGHT / 2 - 20,
                2,
                (255, 100, 100, 255),
            );
            draw_text(
                frame,
                &format!("SCORE: {}", self.score),
                WIDTH / 2 - 70,
                HEIGHT / 2 + 20,
                2,
                (255, 255, 255, 255),
            );
            draw_text(
                frame,
                "PRESS R TO RESTART",
                WIDTH / 2 - 130,
                HEIGHT / 2 + 60,
                2,
                (200, 200, 200, 255),
            );
        } else if self.paused {
            draw_text(
                frame,
                "PAUSED",
                WIDTH / 2 - 50,
                HEIGHT / 2,
                2,
                (255, 255, 100, 255),
            );
        }

        // Note: Score/Length are drawn inside the overlay panel (same plane) in the RedrawRequested block
    }

    /// Fill a single cell-sized rectangle at the given grid position with an RGB color.
    fn draw_rect(&self, frame: &mut [u8], grid_x: u32, grid_y: u32, r: u8, g: u8, b: u8) {
        let x = grid_x * GRID_SIZE;
        let y = grid_y * GRID_SIZE;

        for py in y..y + GRID_SIZE {
            for px in x..x + GRID_SIZE {
                if px < WIDTH && py < HEIGHT {
                    let idx = ((py * WIDTH + px) * 4) as usize;
                    if idx + 3 < frame.len() {
                        frame[idx] = r;
                        frame[idx + 1] = g;
                        frame[idx + 2] = b;
                        frame[idx + 3] = 255;
                    }
                }
            }
        }
    }

    /// Draw simple black "eyes" on the snake head based on current direction.
    fn draw_eyes(&self, frame: &mut [u8], pos: &Pos) {
        let base_x = pos.x as u32 * GRID_SIZE;
        let base_y = pos.y as u32 * GRID_SIZE;

        let (eye1_x, eye1_y, eye2_x, eye2_y) = match self.dir {
            Dir::Right => (base_x + 12, base_y + 5, base_x + 12, base_y + 12),
            Dir::Left => (base_x + 5, base_y + 5, base_x + 5, base_y + 12),
            Dir::Up => (base_x + 5, base_y + 5, base_x + 12, base_y + 5),
            Dir::Down => (base_x + 5, base_y + 12, base_x + 12, base_y + 12),
        };

        blend_pixel(frame, eye1_x, eye1_y, 0, 0, 0, 255);
        blend_pixel(frame, eye2_x, eye2_y, 0, 0, 0, 255);
    }
}

// ============================
// Simple Q-learning Agent (used inside Evolution only)
// ============================

/// Simple Q-learning agent with epsilon-greedy policy.
#[derive(Clone, Serialize, Deserialize)]
struct QAgent {
    q: AHashMap<u32, [f32; 3]>,
    epsilon: f32,
    min_epsilon: f32,
    decay: f32,
    alpha: f32,
    gamma: f32,
    steps: u64,
    episodes: u64,
    #[serde(skip)]
    color: (u8, u8, u8), // RGB цвет агента (не сохраняется)
}

impl QAgent {
    /// Construct a new agent with balanced hyperparameters for the 20-bit state.
    fn new() -> Self {
        // Сбалансированные параметры для 20-битного vision
        // Дефолтный цвет - яркий зелёный (будет перезаписан при создании популяции)
        Self {
            q: AHashMap::new(),
            epsilon: 0.25,
            min_epsilon: 0.05,
            decay: 0.9992,
            alpha: 0.3,
            gamma: 0.95,
            steps: 0,
            episodes: 0,
            color: (100, 220, 100),
        }
    }

    /// Construct an agent and set its display color.
    fn new_with_color(r: u8, g: u8, b: u8) -> Self {
        let mut agent = Self::new();
        agent.color = (r, g, b);
        agent
    }

    /// Get or initialize the Q-values array for a state key.
    fn get_qs(&mut self, s: u32) -> &mut [f32; 3] {
        self.q.entry(s).or_insert([0.0, 0.0, 0.0])
    }

    /// Select an action index {0:left, 1:straight, 2:right} using epsilon-greedy policy.
    fn select_action<R: Rng + ?Sized>(&mut self, s: u32, rng: &mut R) -> usize {
        if rng.r#gen::<f32>() < self.epsilon {
            rng.gen_range(0..3)
        } else {
            let qs = *self.get_qs(s);
            if qs[0] >= qs[1] && qs[0] >= qs[2] {
                0
            } else if qs[1] >= qs[2] {
                1
            } else {
                2
            }
        }
    }

    /// Q-learning update for (state, action, reward, next_state, done).
    fn learn(&mut self, s: u32, a: usize, r: f32, ns: u32, done: bool) {
        let next_max = if done {
            0.0
        } else {
            let nqs = self.q.get(&ns).copied().unwrap_or([0.0; 3]);
            nqs[0].max(nqs[1]).max(nqs[2])
        };
        let alpha = self.alpha;
        let gamma = self.gamma;
        let qsa = self.get_qs(s);
        let td_target = r + gamma * next_max;
        qsa[a] = qsa[a] + alpha * (td_target - qsa[a]);
    }

    // Reset exploration parameters for more aggressive learning
    /// Temporarily increase exploration and learning rate (used on restarts).
    fn boost_exploration(&mut self) {
        self.epsilon = 0.35; // умеренное увеличение
        self.alpha = 0.45; // умеренное ускорение обучения
    }
}

// ============================
// Evolutionary trainer (population of agents)
// ============================

/// Evolutionary trainer managing a population of QAgents and parallel games.
struct EvoTrainer {
    training: bool,
    solved: bool,
    pop: Vec<QAgent>,
    pop_size: usize,
    current: usize,
    epoch: usize,
    epoch_best: Vec<usize>,
    scores: Vec<usize>,
    step_limit: u32,
    steps_taken: u32,
    target_score: usize,
    best_score: usize,
    games: Vec<Game>,                  // parallel games for each individual
    champion: Option<QAgent>,          // best agent ever found
    champion_score: usize,             // best score ever achieved
    champion_epoch: usize,             // epoch when champion was found
    epochs_without_improvement: usize, // counter for stagnation
    restart_count: usize,              // number of restarts performed
}

impl EvoTrainer {
    /// Create a trainer with `pop_size` agents and parallel games.
    fn new(pop_size: usize) -> Self {
        let mut pop = Vec::with_capacity(pop_size);
        let mut games = Vec::with_capacity(pop_size);

        // Генерируем уникальные цвета для каждого агента в популяции
        let colors = generate_population_colors(pop_size);

        for &(r, g, b) in colors.iter().take(pop_size) {
            pop.push(QAgent::new_with_color(r, g, b));
            games.push(Game::new());
        }
        let max_apples = (GRID_WIDTH as usize * GRID_HEIGHT as usize).saturating_sub(3); // 3 is initial snake length
        Self {
            training: false,
            solved: false,
            pop,
            pop_size,
            current: 0,
            epoch: 0,
            epoch_best: Vec::new(),
            scores: vec![0; pop_size],
            step_limit: 4000,
            steps_taken: 0,
            target_score: max_apples,
            best_score: 0,
            games,
            champion: None,
            champion_score: 0,
            champion_epoch: 0,
            epochs_without_improvement: 0,
            restart_count: 0,
        }
    }

    /// Save the current champion (or best of population) to JSON.
    fn save_best(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Save the champion if we have one, otherwise save current best
        let agent_to_save = if let Some(ref champ) = self.champion {
            champ
        } else if !self.pop.is_empty() {
            let mut idxs: Vec<usize> = (0..self.pop_size).collect();
            idxs.sort_by_key(|&i| std::cmp::Reverse(self.scores[i]));
            &self.pop[*idxs.first().unwrap_or(&0)]
        } else {
            return Ok(());
        };
        let json = serde_json::to_string_pretty(agent_to_save)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load a champion agent from JSON and seed the population from it.
    fn load_best(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        if !Path::new(path).exists() {
            return Ok(());
        }
        let json = fs::read_to_string(path)?;
        let agent: QAgent = serde_json::from_str(&json)?;

        // Генерируем яркие цвета для загруженных агентов
        let colors = generate_population_colors(self.pop_size);

        // Replace all agents with the loaded one + assign colors
        for (p, &color) in self.pop.iter_mut().zip(colors.iter()).take(self.pop_size) {
            *p = agent.clone();
            p.color = color; // устанавливаем уникальный цвет
        }
        Ok(())
    }

    /// Reset per-epoch counters and restart all games.
    fn reset_epoch(&mut self) {
        self.current = 0;
        self.steps_taken = 0;
        self.scores.fill(0);
        for i in 0..self.pop_size {
            self.games[i] = Game::new();
        }
    }

    /// Reproduce a new generation with elitism, mutation, and adaptive restarts.
    fn reproduce<R: Rng + ?Sized>(&mut self, rng: &mut R, save_path: &str) {
        let mut idxs: Vec<usize> = (0..self.pop_size).collect();
        idxs.sort_by_key(|&i| std::cmp::Reverse(self.scores[i]));
        let best_idx = *idxs.first().unwrap_or(&0);
        let best_score = self.scores[best_idx];
        self.epoch_best.push(best_score);

        self.best_score = self.best_score.max(best_score);

        let mut new_champion = false;
        // Update global champion if this is a new record
        if best_score > self.champion_score {
            self.champion_score = best_score;
            self.champion_epoch = self.epoch;
            self.champion = Some(self.pop[best_idx].clone());
            self.epochs_without_improvement = 0; // reset stagnation counter
            new_champion = true;
            println!(
                "🏆 NEW CHAMPION! Score: {} (Epoch {})",
                best_score, self.epoch
            );

            // Auto-save immediately when new champion found
            if let Err(e) = self.save_best(save_path) {
                eprintln!("Failed to save champion: {}", e);
            } else {
                println!("✅ Champion saved to {}", save_path);
            }
        } else {
            self.epochs_without_improvement += 1;
        }

        let mut new_pop: Vec<QAgent> = Vec::with_capacity(self.pop_size);

        // Adaptive stagnation threshold: increase after each restart to give more time
        let base_threshold = 1000;
        let stagnation_threshold = base_threshold + (self.restart_count * 500);

        // Check for long stagnation
        if self.epochs_without_improvement >= stagnation_threshold && self.champion.is_some() {
            // After 5 restarts, cycle back to restart #1 but with even more aggressive exploration
            if self.restart_count >= 5 {
                self.restart_count = 0; // cycle back
                println!("🔄 Max restarts reached. Cycling back with aggressive exploration...");
            }

            self.restart_count += 1;
            println!(
                "⚠️ Stagnation detected ({} epochs without improvement). Restart #{} with exploration...",
                self.epochs_without_improvement, self.restart_count
            );
            self.epochs_without_improvement = 0; // reset counter

            // Multi-strategy restart based on restart count
            let champion = self.champion.as_ref().unwrap();

            match self.restart_count {
                1 => {
                    // First restart: moderate mutation + boost exploration
                    new_pop.push(champion.clone());
                    while new_pop.len() < self.pop_size {
                        let mut child = champion.clone();
                        child.boost_exploration(); // reset epsilon and alpha
                        mutate_qagent(&mut child, rng, 0.25); // moderate mutation
                        // Slightly mutate color for diversity
                        child.color = mutate_color(champion.color, 20);
                        new_pop.push(child);
                    }
                }
                2 => {
                    // Second restart: high mutation + more fresh agents + boost
                    new_pop.push(champion.clone());
                    for _ in 1..(self.pop_size / 2) {
                        // changed from 2/3 to 1/2
                        let mut child = champion.clone();
                        child.boost_exploration();
                        mutate_qagent(&mut child, rng, 0.4); // high mutation
                        child.color = mutate_color(champion.color, 30);
                        new_pop.push(child);
                    }
                    // Add more fresh random agents (50%) with new colors
                    let remaining = self.pop_size - new_pop.len();
                    let new_colors = generate_population_colors(remaining);
                    for &color in new_colors.iter() {
                        let mut agent = QAgent::new();
                        agent.color = color;
                        new_pop.push(agent);
                    }
                }
                3 => {
                    // Third restart: 30% champion, 70% fresh agents
                    new_pop.push(champion.clone());
                    for _ in 1..(self.pop_size * 3 / 10) {
                        // 30%
                        let mut child = champion.clone();
                        child.boost_exploration();
                        mutate_qagent(&mut child, rng, 0.35);
                        child.color = mutate_color(champion.color, 40);
                        new_pop.push(child);
                    }
                    // Add fresh random agents (70%) with new colors
                    let remaining = self.pop_size - new_pop.len();
                    let new_colors = generate_population_colors(remaining);
                    for &color in new_colors.iter() {
                        let mut agent = QAgent::new();
                        agent.color = color;
                        new_pop.push(agent);
                    }
                }
                4 => {
                    // Fourth restart: 20% champion + 80% fresh agents + boost
                    new_pop.push(champion.clone());
                    for _ in 1..(self.pop_size / 5) {
                        // 20%
                        let mut child = champion.clone();
                        child.boost_exploration();
                        mutate_qagent(&mut child, rng, 0.6); // very high mutation
                        child.color = mutate_color(champion.color, 50);
                        new_pop.push(child);
                    }
                    // Add mostly fresh random agents (80%) with new colors
                    let remaining = self.pop_size - new_pop.len();
                    let new_colors = generate_population_colors(remaining);
                    for &color in new_colors.iter() {
                        let mut agent = QAgent::new();
                        agent.color = color;
                        new_pop.push(agent);
                    }
                }
                _ => {
                    // Fifth restart: 10% champion + 90% fresh agents + extreme boost
                    new_pop.push(champion.clone());
                    for _ in 1..(self.pop_size / 10) {
                        // 10%
                        let mut child = champion.clone();
                        child.boost_exploration();
                        mutate_qagent(&mut child, rng, 0.8); // extreme mutation
                        child.color = mutate_color(champion.color, 60);
                        new_pop.push(child);
                    }
                    // Add mostly fresh random agents (90%) with new colors + boost
                    let remaining = self.pop_size - new_pop.len();
                    let new_colors = generate_population_colors(remaining);
                    for &color in new_colors.iter() {
                        let mut agent = QAgent::new();
                        agent.boost_exploration(); // boost fresh agents too
                        agent.color = color;
                        new_pop.push(agent);
                    }
                }
            }
        }
        // If we have a new champion, restart population from champion's children
        else if new_champion && self.champion.is_some() {
            self.restart_count = 0; // reset restart counter on new champion
            let champion = self.champion.as_ref().unwrap();
            // First agent is the champion itself (elitism)
            new_pop.push(champion.clone());
            // Rest are mutated versions of the champion with slight color variations
            while new_pop.len() < self.pop_size {
                let mut child = champion.clone();
                mutate_qagent(&mut child, rng, 0.15); // moderate mutation for exploration
                child.color = mutate_color(champion.color, 25); // slight color variation
                new_pop.push(child);
            }
        } else {
            // Normal reproduction: 3 элиты + 4 детей + 3 новых (баланс эксплуатации и исследования)
            let top_k = 3.min(self.pop_size);

            // 1. Elitism: keep top 3 unchanged (30%)
            for &idx in idxs.iter().take(top_k) {
                new_pop.push(self.pop[idx].clone());
            }

            // 2. Создаём 4 детей от элиты с мутациями и смешением цветов (40%)
            let num_children = 4;
            for _ in 0..num_children {
                // Выбираем двух случайных родителей из топ-3
                let parent1_idx = idxs[rng.gen_range(0..top_k)];
                let parent2_idx = idxs[rng.gen_range(0..top_k)];

                let mut child = self.pop[parent1_idx].clone();

                // Умеренная мутация Q-таблицы
                mutate_qagent(&mut child, rng, 0.15);

                // Смешиваем цвета родителей для визуального наследования
                let ratio = rng.gen_range(0.3..0.7);
                let c1 = self.pop[parent1_idx].color;
                let c2 = self.pop[parent2_idx].color;
                let blended = (
                    ((c1.0 as f32 * (1.0 - ratio) + c2.0 as f32 * ratio) as u8),
                    ((c1.1 as f32 * (1.0 - ratio) + c2.1 as f32 * ratio) as u8),
                    ((c1.2 as f32 * (1.0 - ratio) + c2.2 as f32 * ratio) as u8),
                );

                // Добавляем небольшую мутацию цвета для уникальности каждого ребёнка
                child.color = mutate_color(blended, 15);

                new_pop.push(child);
            }

            // 3. Добавляем 3 новых случайных агента с уникальными цветами (30%)
            let num_fresh = 3.min(self.pop_size - new_pop.len());
            let fresh_colors = generate_population_colors(num_fresh);

            for &color in fresh_colors.iter().take(num_fresh) {
                let mut agent = QAgent::new();
                agent.color = color;
                new_pop.push(agent);
            }

            // 4. Дозаполняем популяцию до целевого размера
            if new_pop.len() < self.pop_size {
                let remaining = self.pop_size - new_pop.len();
                let extra_colors = generate_population_colors(remaining);
                for &color in extra_colors.iter().take(remaining) {
                    let mut agent = QAgent::new();
                    agent.color = color;
                    new_pop.push(agent);
                }
            } else if new_pop.len() > self.pop_size {
                new_pop.truncate(self.pop_size);
            }
        }

        self.pop = new_pop;
        self.epoch += 1;
        self.reset_epoch();
    }
}

/// Mutate Q-values and decay epsilon slightly; `sigma` controls noise magnitude.
fn mutate_qagent<R: Rng + ?Sized>(agent: &mut QAgent, rng: &mut R, sigma: f32) {
    for arr in agent.q.values_mut() {
        for v in arr.iter_mut() {
            *v += rng.gen_range(-sigma..sigma);
        }
    }
    agent.epsilon = (agent.epsilon * agent.decay).max(agent.min_epsilon);
}

/// Rotate direction 90° left.
fn left_dir(d: Dir) -> Dir {
    match d {
        Dir::Up => Dir::Left,
        Dir::Left => Dir::Down,
        Dir::Down => Dir::Right,
        Dir::Right => Dir::Up,
    }
}
/// Rotate direction 90° right.
fn right_dir(d: Dir) -> Dir {
    match d {
        Dir::Up => Dir::Right,
        Dir::Right => Dir::Down,
        Dir::Down => Dir::Left,
        Dir::Left => Dir::Up,
    }
}
/// Apply an action index to a direction: 0=left, 1=straight, 2=right.
fn dir_after_action(d: Dir, a: usize) -> Dir {
    match a {
        0 => left_dir(d),
        1 => d,
        _ => right_dir(d),
    }
}

// Генерирует разнообразные цвета для популяции
/// Generate distinct RGB colors for a population using HSL hue sampling.
fn generate_population_colors(pop_size: usize) -> Vec<(u8, u8, u8)> {
    let mut colors = Vec::with_capacity(pop_size);
    for i in 0..pop_size {
        let hue = (i as f32 / pop_size as f32) * 360.0;
        let (r, g, b) = hsl_to_rgb(hue, 0.85, 0.65); // увеличена насыщенность и яркость
        colors.push((r, g, b));
    }
    colors
}

// Конвертирует HSL в RGB
/// Convert HSL to RGB (0..=255 per channel).
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (u8, u8, u8) {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let h_prime = h / 60.0;
    let x = c * (1.0 - ((h_prime % 2.0) - 1.0).abs());
    let (r1, g1, b1) = match h_prime as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        5 => (c, 0.0, x),
        _ => (c, x, 0.0),
    };
    let m = l - c / 2.0;
    (
        ((r1 + m) * 255.0) as u8,
        ((g1 + m) * 255.0) as u8,
        ((b1 + m) * 255.0) as u8,
    )
}

// Мутирует цвет с небольшим изменением
/// Slightly mutate an RGB color by ±`range` per channel (clamped to 0..255).
fn mutate_color(color: (u8, u8, u8), range: i32) -> (u8, u8, u8) {
    let mut rng: SmallRng = SmallRng::from_entropy();
    let r = (color.0 as i32 + rng.gen_range(-range..=range)).clamp(0, 255) as u8;
    let g = (color.1 as i32 + rng.gen_range(-range..=range)).clamp(0, 255) as u8;
    let b = (color.2 as i32 + rng.gen_range(-range..=range)).clamp(0, 255) as u8;
    (r, g, b)
}

/// Build a compact 20-bit state key from the game: 16 bits of local vision, 2 bits
/// of relative direction to the apple, and 2 bits of distance bucket.
fn state_key(game: &Game) -> u32 {
    // Компактный vision-based подход БЕЗ хэширования
    // Смотрим только на критически важные клетки вокруг головы (3x3 впереди)
    // Итого: 16 бит для vision + 4 бита для контекста = 20 бит (~1M состояний)

    let head = game.snake.front().unwrap();
    let mut k: u32 = 0;

    // Получаем 8 клеток вокруг головы относительно направления движения
    // Кодируем каждую клетку 2 битами: 00=пусто, 01=опасность(стена/тело), 10=яблоко, 11=unused
    let checks = [
        (-1, -1),
        (0, -1),
        (1, -1), // left-ahead, ahead, right-ahead (приоритетные)
        (-1, 0),
        (1, 0), // left, right
        (-1, 1),
        (0, 1),
        (1, 1), // left-behind, behind, right-behind
    ];

    let mut bit_pos = 0;
    for (dx, dy) in &checks {
        // Преобразуем относительные координаты в зависимости от направления
        let (world_dx, world_dy) = match game.dir {
            Dir::Right => (*dy, *dx),
            Dir::Left => (-*dy, -*dx),
            Dir::Up => (*dx, -*dy),
            Dir::Down => (-*dx, *dy),
        };

        let check_x = head.x + world_dx;
        let check_y = head.y + world_dy;

        let cell_value = if check_x < 0
            || check_x >= GRID_WIDTH as i32
            || check_y < 0
            || check_y >= GRID_HEIGHT as i32
        {
            1 // стена/граница = опасность
        } else {
            let pos = Pos::new(check_x, check_y);
            if game.snake_set.contains(&pos) {
                1 // тело змеи = опасность
            } else if pos == game.apple {
                2 // яблоко
            } else {
                0 // пусто
            }
        };

        k |= (cell_value as u32) << bit_pos;
        bit_pos += 2;
    }

    // Биты 16-17: направление к яблоку (left/straight/right относительно текущего направления)
    let apple_dx = game.apple.x - head.x;
    let apple_dy = game.apple.y - head.y;
    let apple_dir = match game.dir {
        Dir::Right => {
            if apple_dy < -1 {
                0
            }
            // left
            else if apple_dy > 1 {
                2
            }
            // right
            else {
                1
            } // straight-ish
        }
        Dir::Left => {
            if apple_dy > 1 {
                0
            }
            // left
            else if apple_dy < -1 {
                2
            }
            // right
            else {
                1
            } // straight-ish
        }
        Dir::Up => {
            if apple_dx < -1 {
                0
            }
            // left
            else if apple_dx > 1 {
                2
            }
            // right
            else {
                1
            } // straight-ish
        }
        Dir::Down => {
            if apple_dx > 1 {
                0
            }
            // left
            else if apple_dx < -1 {
                2
            }
            // right
            else {
                1
            } // straight-ish
        }
    };
    k |= apple_dir << 16;

    // Биты 18-19: дистанция до яблока (4 категории)
    let dist = apple_dx.abs() + apple_dy.abs();
    let dist_cat = if dist <= 3 {
        0
    } else if dist <= 8 {
        1
    } else if dist <= 16 {
        2
    } else {
        3
    };
    k |= dist_cat << 18;

    k
}

/// Entry point: sets up the window, renderer, input loop, and optionally runs
/// evolutionary training.
fn main() -> Result<(), Error> {
    #[cfg(feature = "gpu-nn")]
    {
        println!("[mode] GPU NN feature enabled (scaffold)");
    }
    #[cfg(feature = "gpu-nn-experimental")]
    {
        println!("[mode] GPU NN experimental backend enabled");
    }
    let event_loop = EventLoop::new();
    let mut input = WinitInputHelper::new();

    let window = WindowBuilder::new()
        .with_title("🐍 Snake Game")
        .with_inner_size(LogicalSize::new(WIDTH, HEIGHT))
        .with_resizable(true) // allow resizing
        .build(&event_loop)
        .unwrap();

    let mut pixels = {
        let window_size = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        Pixels::new(WIDTH, HEIGHT, surface_texture)?
    };

    let mut game = Game::new();
    let mut evo = EvoTrainer::new(24); // увеличенная популяция для более быстрого поиска решений
    #[cfg(feature = "gpu-nn")]
    let mut nn_mode: bool = false;
    #[cfg(feature = "gpu-nn-experimental")]
    let mut nn_trainer: Option<gpu_nn::GpuTrainer> = Some(gpu_nn::GpuTrainer::new(256, 128, 3));

    // Try to load saved agent and auto-start training if found
    let save_path = "snake_agent.json";
    let agent_loaded = if let Err(e) = evo.load_best(save_path) {
        eprintln!("Could not load saved agent: {}", e);
        false
    } else {
        println!("✅ Loaded saved agent from {}", save_path);
        true
    };

    // Auto-start evolution if agent was loaded
    if agent_loaded {
        evo.training = true;
        println!("🚀 Auto-starting evolution with loaded agent");
    }

    let mut rng: SmallRng = SmallRng::from_entropy();
    let mut last_update = Instant::now();
    let mut tick_duration = Duration::from_millis(150);
    let mut manual_speed_delta_ms: i32 = 0;
    let mut evo_steps_per_frame: u32 = 1; // начальная скорость = 1 шаг за кадр (медленно для наблюдения)
    let mut panel_visible: bool = true; // panel visibility toggle
    let mut frame_counter: u32 = 0; // counter for skipping frames
    // Evolution step budget to spread very large step counts across ticks
    let mut evo_pending_steps: u32 = 0;
    let mut max_steps_per_tick: u32 = 1500; // cap work per tick to keep UI responsive
    let mut ultra_fast: bool = false; // training ultra-fast mode (disable render, raise cap)
    let mut show_only_best: bool = false; // render only the best agent during training
    // GPU detection (wgpu) and accel flags
    let mut gpu_available: bool = false;
    let mut gpu_enabled: bool = false;
    {
        let instance = Instance::new(wgpu::InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });
        if let Some(_adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }))
        {
            gpu_available = true;
            gpu_enabled = true;
            max_steps_per_tick = 80_000;
        }
    }
    // FPS counter state
    let mut fps_last: Instant = Instant::now();
    let mut fps_frames: u32 = 0;
    let mut fps_value: f32 = 0.0;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        if let Event::RedrawRequested(_) = event {
            let frame = pixels.frame_mut();

            // Draw the appropriate game(s)
            if evo.training {
                // Rendering strategy tuned for performance at high EVO speeds
                if ultra_fast {
                    // Ultra-fast: no render during training
                    clear_rgba(frame, 10, 10, 15, 255);
                } else if show_only_best {
                    // Always render only the best agent
                    clear_rgba(frame, 10, 10, 15, 255);
                    if let Some(best_game_idx) = evo
                        .scores
                        .iter()
                        .enumerate()
                        .max_by_key(|(_, score)| *score)
                        .map(|(idx, _)| idx)
                        && best_game_idx < evo.pop.len()
                        && best_game_idx < evo.games.len()
                    {
                        let agent_color = evo.pop[best_game_idx].color;
                        draw_game_transparent(frame, &evo.games[best_game_idx], 220, agent_color);
                    }
                } else if evo_steps_per_frame < 8_192 {
                    // Low/medium speed: draw grid + agents
                    clear_rgba(frame, 30, 30, 40, 255);
                    // Draw grid first
                    for y in 0..GRID_HEIGHT {
                        for x in 0..GRID_WIDTH {
                            if (x + y) % 2 == 0 {
                                let gx = x * GRID_SIZE;
                                let gy = y * GRID_SIZE;
                                for py in gy..gy + GRID_SIZE {
                                    for px in gx..gx + GRID_SIZE {
                                        if px < WIDTH && py < HEIGHT {
                                            let idx = ((py * WIDTH + px) * 4) as usize;
                                            if idx + 3 < frame.len() {
                                                frame[idx] = 35;
                                                frame[idx + 1] = 35;
                                                frame[idx + 2] = 50;
                                                frame[idx + 3] = 255;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // Draw all individuals only when really slow; otherwise only best
                    if evo_steps_per_frame < 4_096 {
                        for (agent, g) in evo.pop.iter().zip(evo.games.iter()) {
                            let agent_color = agent.color;
                            draw_game_transparent(frame, g, 180, agent_color);
                        }
                    } else if let Some(best_game_idx) = evo
                        .scores
                        .iter()
                        .enumerate()
                        .max_by_key(|(_, score)| *score)
                        .map(|(idx, _)| idx)
                        && best_game_idx < evo.pop.len()
                        && best_game_idx < evo.games.len()
                    {
                        let agent_color = evo.pop[best_game_idx].color;
                        draw_game_transparent(frame, &evo.games[best_game_idx], 220, agent_color);
                    }
                } else if evo_steps_per_frame < 20_000 {
                    // High speed: skip grid entirely; draw best only on plain background
                    clear_rgba(frame, 10, 10, 15, 255);
                    if let Some(best_game_idx) = evo
                        .scores
                        .iter()
                        .enumerate()
                        .max_by_key(|(_, score)| *score)
                        .map(|(idx, _)| idx)
                        && best_game_idx < evo.pop.len()
                        && best_game_idx < evo.games.len()
                    {
                        let agent_color = evo.pop[best_game_idx].color;
                        draw_game_transparent(frame, &evo.games[best_game_idx], 220, agent_color);
                    }
                } else {
                    // Ultra-high speed: don't render agents at all
                    clear_rgba(frame, 10, 10, 15, 255);
                }
            } else {
                game.draw(frame);
            }

            // Controls overlay (semi-transparent) - only draw if visible
            if panel_visible {
                let panel_x: u32 = 8;
                let panel_y: u32 = 8;
                let panel_w: u32 = 380; // increased from 280
                let panel_h: u32 = 628; // increased to fit new button line
                let btn_h: u32 = 32; // increased button height
                let btn_w: u32 = panel_w - 16;
                let btn_x: u32 = panel_x + 8;
                // Chart area inside panel (positioned below HUD option lines)
                let chart_y: u32 = panel_y + 340; // moved further down to avoid text overlap
                let chart_h: u32 = 120; // increased chart height
                let btn1_y: u32 = chart_y + chart_h + 8; // start buttons after chart
                let btn2_y: u32 = btn1_y + btn_h + 6;
                let btn3_y: u32 = btn2_y + btn_h + 6;
                let btn4_y: u32 = btn3_y + btn_h + 6;
                let btn5_y: u32 = btn4_y + btn_h + 6; // hide button
                let btn6_y: u32 = btn5_y + btn_h + 6; // show-only-best button

                fill_rect_rgba(frame, panel_x, panel_y, panel_w, panel_h, 0, 0, 0, 140);
                stroke_rect_rgba(frame, panel_x, panel_y, panel_w, panel_h, 255, 255, 255, 60);
                draw_text(
                    frame,
                    "CONTROLS",
                    panel_x + 10,
                    panel_y + 10,
                    2,
                    (180, 220, 255, 255),
                );
                // HUD inside panel with extra line spacing
                draw_text(
                    frame,
                    &format!("SCORE: {}", game.score),
                    panel_x + 10,
                    panel_y + 40,
                    2,
                    (230, 230, 230, 255),
                );
                draw_text(
                    frame,
                    &format!("LENGTH: {}", game.snake.len()),
                    panel_x + 10,
                    panel_y + 70,
                    2,
                    (200, 200, 200, 255),
                );
                #[cfg(feature = "gpu-nn")]
                {
                    let nn_label = if nn_mode { "ON" } else { "OFF" };
                    draw_text(
                        frame,
                        &format!("NN MODE [N]: {}", nn_label),
                        panel_x + 10,
                        panel_y + 100,
                        2,
                        (180, 220, 180, 255),
                    );
                }
                // Speed indicator (based on tick duration)
                let ms = tick_duration.as_millis() as f32;
                let sps = if ms > 0.0 { 1000.0 / ms } else { 0.0 };
                draw_text(
                    frame,
                    &format!("SPEED: {} ms (~{:.1}/s)", ms as i32, sps),
                    panel_x + 10,
                    panel_y + 100,
                    2,
                    (200, 220, 255, 255),
                );

                // Evolutionary training status
                draw_text(
                    frame,
                    &format!("EVO: {} (E)", if evo.training { "ON" } else { "OFF" }),
                    panel_x + 10,
                    panel_y + 130,
                    2,
                    (220, 200, 240, 255),
                );
                let gpu_str = if gpu_enabled {
                    "GPU"
                } else if gpu_available {
                    "GPU avail"
                } else {
                    "CPU"
                };
                draw_text(
                    frame,
                    &format!("ACCEL: {}  (G)", gpu_str),
                    panel_x + 200,
                    panel_y + 130,
                    2,
                    (180, 255, 200, 255),
                );
                let alive_count = evo.games.iter().filter(|g| g.alive).count();
                draw_text(
                    frame,
                    &format!(
                        "EPOCH: {}  ALIVE: {}/{}",
                        evo.epoch, alive_count, evo.pop_size
                    ),
                    panel_x + 10,
                    panel_y + 160,
                    2,
                    (220, 200, 240, 255),
                );
                draw_text(
                    frame,
                    &format!("TARGET: {}  BEST: {}", evo.target_score, evo.best_score),
                    panel_x + 10,
                    panel_y + 190,
                    2,
                    (220, 200, 240, 255),
                );
                // Leader protection HUD: show when unique leader bypasses step limit
                {
                    let (mut top1, mut top2, mut top1_idx) = (0usize, 0usize, None::<usize>);
                    for (i, &sc) in evo.scores.iter().enumerate() {
                        if sc > top1 {
                            top2 = top1;
                            top1 = sc;
                            top1_idx = Some(i);
                        } else if sc > top2 {
                            top2 = sc;
                        }
                    }
                    let leader_protected = if let Some(idx) = top1_idx {
                        (top1 > top2) && evo.games.get(idx).map(|g| g.alive).unwrap_or(false)
                    } else {
                        false
                    };
                    if leader_protected {
                        draw_text(
                            frame,
                            "LEADER: protected",
                            panel_x + 10,
                            panel_y + 220,
                            2,
                            (120, 255, 120, 255),
                        );
                    }
                }

                // Champion info with epoch
                if evo.champion_score > 0 {
                    draw_text(
                        frame,
                        &format!(
                            "CHAMPION: {} (epoch {})",
                            evo.champion_score, evo.champion_epoch
                        ),
                        panel_x + 10,
                        panel_y + 240,
                        2,
                        (255, 215, 0, 255),
                    );
                } else {
                    draw_text(
                        frame,
                        "CHAMPION: None",
                        panel_x + 10,
                        panel_y + 240,
                        2,
                        (255, 215, 0, 255),
                    );
                }

                // Stagnation warning
                if evo.epochs_without_improvement > 0 {
                    let base_threshold = 1000 + (evo.restart_count * 500);
                    let color = if evo.epochs_without_improvement > (base_threshold - 200) {
                        (255, 100, 100, 255)
                    } else {
                        (200, 200, 200, 255)
                    };
                    draw_text(
                        frame,
                        &format!(
                            "No improvement: {}/{} (restarts: {})",
                            evo.epochs_without_improvement, base_threshold, evo.restart_count
                        ),
                        panel_x + 10,
                        panel_y + 270,
                        2,
                        color,
                    );
                }

                let ultra_str = if ultra_fast { "ON" } else { "OFF" };
                let best_str = if show_only_best { "ON" } else { "OFF" };
                // Split into two lines to keep within panel width
                draw_text(
                    frame,
                    &format!("EVO SPD: {} steps/frame (+/-)", evo_steps_per_frame),
                    panel_x + 10,
                    panel_y + 300,
                    2,
                    (200, 220, 255, 255),
                );
                draw_text(
                    frame,
                    &format!("ULTRA (U): {}    BEST (B): {}", ultra_str, best_str),
                    panel_x + 10,
                    panel_y + 320,
                    2,
                    (200, 220, 255, 255),
                );
                // Chart of best apples per epoch
                draw_chart(
                    frame,
                    panel_x + 10,
                    chart_y,
                    panel_w - 20,
                    chart_h,
                    &evo.epoch_best,
                );

                let paused_label = if game.paused {
                    "RESUME  P"
                } else {
                    "PAUSE   P"
                };
                draw_button(frame, btn_x, btn1_y, btn_w, btn_h, paused_label);
                draw_button(frame, btn_x, btn2_y, btn_w, btn_h, "SPEED+  +");
                draw_button(frame, btn_x, btn3_y, btn_w, btn_h, "RESTART R");
                draw_button(frame, btn_x, btn4_y, btn_w, btn_h, "SAVE    S");
                draw_button(frame, btn_x, btn5_y, btn_w, btn_h, "HIDE    H");
                let best_label = if show_only_best {
                    "BEST ON  B"
                } else {
                    "BEST OFF B"
                };
                draw_button(frame, btn_x, btn6_y, btn_w, btn_h, best_label);
            } else {
                // Draw small button to show panel again
                let show_btn_x: u32 = 8;
                let show_btn_y: u32 = 8;
                let show_btn_w: u32 = 100;
                let show_btn_h: u32 = 32;
                draw_button(
                    frame, show_btn_x, show_btn_y, show_btn_w, show_btn_h, "SHOW H",
                );
            }

            // Update and draw FPS counter (top-right, green)
            fps_frames = fps_frames.wrapping_add(1);
            let elapsed = fps_last.elapsed();
            if elapsed.as_secs_f32() >= 1.0 {
                fps_value = fps_frames as f32 / elapsed.as_secs_f32();
                fps_frames = 0;
                fps_last = Instant::now();
            }
            let fps_text = format!("FPS: {:.0}", fps_value);
            let scale: u32 = 2;
            let advance = 5 * scale + scale; // glyph width + spacing
            let text_w: u32 = fps_text.chars().count() as u32 * advance;
            let fps_x: u32 = WIDTH.saturating_sub(text_w + 8);
            let fps_y: u32 = 8;
            draw_text(frame, &fps_text, fps_x, fps_y, scale, (80, 255, 120, 255));

            if pixels.render().is_err() {
                *control_flow = ControlFlow::Exit;
            }
        }

        // Handle window resize
        if let Event::WindowEvent { event, .. } = &event
            && let winit::event::WindowEvent::Resized(new_size) = event
            && let Err(e) = pixels.resize_surface(new_size.width, new_size.height)
        {
            eprintln!("Failed to resize surface: {}", e);
            *control_flow = ControlFlow::Exit;
            return;
        }

        if input.update(&event) {
            // Handle quit
            if input.key_pressed(VirtualKeyCode::Escape)
                || input.close_requested()
                || input.destroyed()
            {
                *control_flow = ControlFlow::Exit;
                return;
            }

            // Handle restart
            if input.key_pressed(VirtualKeyCode::R) && !game.alive {
                game = Game::new();
                tick_duration = Duration::from_millis(150);
            }

            // Handle pause
            if input.key_pressed(VirtualKeyCode::P) {
                game.paused = !game.paused;
            }

            // Evolution toggle only
            if input.key_pressed(VirtualKeyCode::E) {
                evo.training = !evo.training;
                if evo.training {
                    evo.solved = false;
                    evo.reset_epoch();
                    evo.epoch = 0;
                    evo.epoch_best.clear();
                    evo.best_score = 0;
                    evo.epochs_without_improvement = 0;
                    game = Game::new();
                }
            }

            // Save agent
            if input.key_pressed(VirtualKeyCode::S) {
                if let Err(e) = evo.save_best(save_path) {
                    eprintln!("Failed to save agent: {}", e);
                } else {
                    println!("Agent saved to {}", save_path);
                }
            }

            // Toggle panel visibility
            if input.key_pressed(VirtualKeyCode::H) {
                panel_visible = !panel_visible;
            }
            #[cfg(feature = "gpu-nn")]
            {
                if input.key_pressed(VirtualKeyCode::N) {
                    nn_mode = !nn_mode;
                    if nn_mode {
                        println!("[gpu-nn] Enabled NN mode (experimental)");
                    } else {
                        println!("[gpu-nn] Disabled NN mode");
                    }
                }
            }
            // Ultra-fast toggle
            if input.key_pressed(VirtualKeyCode::U) {
                ultra_fast = !ultra_fast;
                max_steps_per_tick = if ultra_fast { 50_000 } else { 1500 };
            }
            // Toggle GPU acceleration mode (just adjusts training budget for now)
            if input.key_pressed(VirtualKeyCode::G) && gpu_available {
                gpu_enabled = !gpu_enabled;
                max_steps_per_tick = if gpu_enabled {
                    80_000
                } else if ultra_fast {
                    50_000
                } else {
                    1500
                };
            }
            if input.key_pressed(VirtualKeyCode::B) {
                show_only_best = !show_only_best;
            }

            // Speed controls (keyboard)
            if evo.training {
                if input.key_pressed(VirtualKeyCode::NumpadAdd)
                    || input.key_pressed(VirtualKeyCode::Equals)
                {
                    evo_steps_per_frame = (evo_steps_per_frame.saturating_mul(2)).min(100_000); // increased max from 10_000 to 100_000
                }
                if input.key_pressed(VirtualKeyCode::NumpadSubtract)
                    || input.key_pressed(VirtualKeyCode::Minus)
                {
                    evo_steps_per_frame = (evo_steps_per_frame / 2).max(1);
                }
            } else {
                if input.key_pressed(VirtualKeyCode::NumpadAdd)
                    || input.key_pressed(VirtualKeyCode::Equals)
                {
                    manual_speed_delta_ms = (manual_speed_delta_ms - 10).max(-150);
                }
                if input.key_pressed(VirtualKeyCode::NumpadSubtract)
                    || input.key_pressed(VirtualKeyCode::Minus)
                {
                    manual_speed_delta_ms = (manual_speed_delta_ms + 10).min(300);
                }
            }

            // Handle direction changes
            if input.key_pressed(VirtualKeyCode::Up) || input.key_pressed(VirtualKeyCode::W) {
                game.change_dir(Dir::Up);
            }
            if input.key_pressed(VirtualKeyCode::Down) || input.key_pressed(VirtualKeyCode::S) {
                game.change_dir(Dir::Down);
            }
            if input.key_pressed(VirtualKeyCode::Left) || input.key_pressed(VirtualKeyCode::A) {
                game.change_dir(Dir::Left);
            }
            if input.key_pressed(VirtualKeyCode::Right) || input.key_pressed(VirtualKeyCode::D) {
                game.change_dir(Dir::Right);
            }

            // Mouse clicks on overlay buttons
            if let Some((mx, my)) = input.mouse()
                && input.mouse_pressed(0)
            {
                let mx = mx as u32;
                let my = my as u32;

                if panel_visible {
                    let panel_x: u32 = 8;
                    let panel_y: u32 = 8;
                    let panel_w: u32 = 380;
                    let btn_h: u32 = 32;
                    let btn_w: u32 = panel_w - 16;
                    let btn_x: u32 = panel_x + 8;
                    let chart_y: u32 = panel_y + 310;
                    let chart_h: u32 = 120;
                    let btn1_y: u32 = chart_y + chart_h + 8;
                    let btn2_y: u32 = btn1_y + btn_h + 6;
                    let btn3_y: u32 = btn2_y + btn_h + 6;
                    let btn4_y: u32 = btn3_y + btn_h + 6;
                    let btn5_y: u32 = btn4_y + btn_h + 6;
                    let btn6_y: u32 = btn5_y + btn_h + 6;
                    if point_in_rect(mx, my, btn_x, btn1_y, btn_w, btn_h) {
                        game.paused = !game.paused;
                    } else if point_in_rect(mx, my, btn_x, btn2_y, btn_w, btn_h) {
                        if evo.training {
                            evo_steps_per_frame =
                                (evo_steps_per_frame.saturating_mul(2)).min(100_000);
                        }
                        // increased max from 10_000
                        else {
                            manual_speed_delta_ms = (manual_speed_delta_ms - 10).max(-150);
                        }
                    } else if point_in_rect(mx, my, btn_x, btn3_y, btn_w, btn_h) {
                        game = Game::new();
                        tick_duration = Duration::from_millis(150);
                    } else if point_in_rect(mx, my, btn_x, btn4_y, btn_w, btn_h) {
                        if let Err(e) = evo.save_best(save_path) {
                            eprintln!("Failed to save agent: {}", e);
                        } else {
                            println!("Agent saved to {}", save_path);
                        }
                    } else if point_in_rect(mx, my, btn_x, btn5_y, btn_w, btn_h) {
                        panel_visible = false;
                    } else if point_in_rect(mx, my, btn_x, btn6_y, btn_w, btn_h) {
                        show_only_best = !show_only_best;
                    }
                } else {
                    // Check if clicked on show button
                    let show_btn_x: u32 = 8;
                    let show_btn_y: u32 = 8;
                    let show_btn_w: u32 = 100;
                    let show_btn_h: u32 = 32;
                    if point_in_rect(mx, my, show_btn_x, show_btn_y, show_btn_w, show_btn_h) {
                        panel_visible = true;
                    }
                }
            }

            // Evolutionary training loop (population of agents - parallelized with rayon)
            if evo.training {
                let steps_per_frame: u32 = evo_steps_per_frame.max(1);
                if game.paused {
                    window.request_redraw();
                    return;
                }

                // Accumulate desired work and process in chunks to avoid long UI stalls
                evo_pending_steps = evo_pending_steps.saturating_add(steps_per_frame);
                let to_run = evo_pending_steps.min(max_steps_per_tick);
                let mut ran_steps: u32 = 0;
                for _ in 0..to_run {
                    let mut all_done = true;
                    let target_score = evo.target_score;
                    let len = evo.pop.len().min(evo.games.len()).min(evo.scores.len());
                    // Parallel iterate zipped mutable slices safely
                    let (pop_slice, _) = evo.pop.split_at_mut(len);
                    let (games_slice, _) = evo.games.split_at_mut(len);
                    let (scores_slice, _) = evo.scores.split_at_mut(len);
                    let solved_flag = AtomicBool::new(false);

                    pop_slice
                        .par_iter_mut()
                        .zip(games_slice.par_iter_mut())
                        .zip(scores_slice.par_iter_mut())
                        .for_each(|((agent, g), score_ref)| {
                            if !g.alive || *score_ref >= target_score {
                                return;
                            }
                            // local RNG per thread (SmallRng)
                            let mut local_rng = SmallRng::from_entropy();
                            let s = state_key(g);
                            let a_idx = agent.select_action(s, &mut local_rng);
                            g.change_dir(dir_after_action(g.dir, a_idx));
                            let before_score = g.score;
                            let was_alive = g.alive;
                            let head0 = *g.snake.front().unwrap();
                            let d0 = (g.apple.x - head0.x).abs() + (g.apple.y - head0.y).abs();
                            g.update();
                            let ate = g.score > before_score;
                            let died = was_alive && !g.alive;
                            let head1 = *g.snake.front().unwrap();
                            let d1 = (g.apple.x - head1.x).abs() + (g.apple.y - head1.y).abs();
                            let length1 = g.snake.len();

                            // Death penalties: self-collision is the heaviest crime
                            let mut reward = if died {
                                match g.last_death {
                                    DeathCause::SelfCollision => -30.0,
                                    DeathCause::None => -12.0,
                                }
                            } else if ate {
                                10.0 + (length1 as f32 * 0.1)
                            } else {
                                -0.005
                            };
                            if !died && !ate {
                                if d1 < d0 {
                                    reward += 0.05;
                                } else if d1 > d0 {
                                    reward -= 0.03;
                                }
                                if d1 <= 3 && !ate {
                                    reward += 0.02;
                                }
                            }

                            let ns = state_key(g);
                            agent.learn(s, a_idx, reward, ns, died || !g.alive);
                            agent.steps += 1;
                            if died {
                                agent.episodes += 1;
                                agent.epsilon =
                                    (agent.epsilon * agent.decay).max(agent.min_epsilon);
                            }
                            if g.alive {
                                *score_ref = g.score;
                            }
                            if g.score >= target_score {
                                solved_flag.store(true, Ordering::Relaxed);
                            }
                        });

                    if solved_flag.load(Ordering::Relaxed) {
                        evo.solved = true;
                        evo.training = false;
                    } else if scores_slice
                        .iter()
                        .zip(games_slice.iter())
                        .any(|(s, g)| g.alive && *s < target_score)
                    {
                        all_done = false;
                    }

                    // Determine if there is a unique leading agent who should bypass the step limit
                    let (mut top1, mut top2, mut top1_idx) = (0usize, 0usize, None::<usize>);
                    for (i, &sc) in evo.scores.iter().enumerate() {
                        if sc > top1 {
                            top2 = top1;
                            top1 = sc;
                            top1_idx = Some(i);
                        } else if sc > top2 {
                            top2 = sc;
                        }
                    }
                    let leader_protected = if let Some(idx) = top1_idx {
                        (top1 > top2) && evo.games.get(idx).map(|g| g.alive).unwrap_or(false)
                    } else {
                        false
                    };

                    evo.steps_taken += 1;
                    ran_steps += 1;
                    if all_done || (evo.steps_taken >= evo.step_limit && !leader_protected) {
                        // All individuals finished or step limit reached - start new epoch
                        evo.reproduce(&mut rng, save_path);
                        evo_pending_steps = 0; // reset pending work on epoch change
                        break;
                    }
                }
                // Reduce pending work by the amount actually processed
                evo_pending_steps = evo_pending_steps.saturating_sub(ran_steps);

                // Update screen less frequently on high speeds to improve performance
                frame_counter += 1;
                let frames_to_skip = if ultra_fast {
                    8
                } else if evo_steps_per_frame >= 65_536 {
                    20 // update screen every 20th iteration
                } else if evo_steps_per_frame >= 40_000 {
                    12 // every 12th
                } else if evo_steps_per_frame >= 20_000 {
                    6 // every 6th
                } else if evo_steps_per_frame >= 8_192 {
                    3 // every 3rd
                } else if evo_steps_per_frame >= 4_096 {
                    2 // every 2nd
                } else {
                    1 // update every iteration
                };

                if !ultra_fast && frame_counter >= frames_to_skip {
                    frame_counter = 0;
                    window.request_redraw();
                }
                return;
            }

            // (Removed) standalone Q-learning training loop

            // Update game logic (real-time); manual play only outside evolution
            if last_update.elapsed() >= tick_duration {
                game.update();
                last_update = Instant::now();

                // Combine base speed with manual delta
                let base_ms = 150 - game.score.min(30) as i32 * 4;
                let total_ms = (base_ms + manual_speed_delta_ms).clamp(30, 500) as u64;
                tick_duration = Duration::from_millis(total_ms);
            }

            window.request_redraw();
        }
    });
}

// ============================
// Rendering helpers and UI
// ============================

/// Clear the entire frame buffer to a single RGBA color.
fn clear_rgba(frame: &mut [u8], r: u8, g: u8, b: u8, a: u8) {
    for px in frame.chunks_exact_mut(4) {
        px[0] = r;
        px[1] = g;
        px[2] = b;
        px[3] = a;
    }
}

/// Alpha-blend a pixel into the frame at (x,y).
fn blend_pixel(frame: &mut [u8], x: u32, y: u32, r: u8, g: u8, b: u8, a: u8) {
    if x >= WIDTH || y >= HEIGHT {
        return;
    }
    let idx = ((y * WIDTH + x) * 4) as usize;
    if idx + 3 >= frame.len() {
        return;
    }
    let ar = a as u16;
    let iar = (255 - a) as u16;
    let dr = frame[idx] as u16;
    let dg = frame[idx + 1] as u16;
    let db = frame[idx + 2] as u16;
    frame[idx] = (((r as u16) * ar + dr * iar) / 255) as u8;
    frame[idx + 1] = (((g as u16) * ar + dg * iar) / 255) as u8;
    frame[idx + 2] = (((b as u16) * ar + db * iar) / 255) as u8;
    frame[idx + 3] = 255;
}

/// Fill an axis-aligned rectangle with an RGBA color (alpha-blended per pixel).
#[allow(clippy::too_many_arguments)]
fn fill_rect_rgba(frame: &mut [u8], x: u32, y: u32, w: u32, h: u32, r: u8, g: u8, b: u8, a: u8) {
    let x2 = (x + w).min(WIDTH);
    let y2 = (y + h).min(HEIGHT);
    for py in y..y2 {
        for px in x..x2 {
            blend_pixel(frame, px, py, r, g, b, a);
        }
    }
}

/// Draw a rectangle border with an RGBA color.
#[allow(clippy::too_many_arguments)]
fn stroke_rect_rgba(frame: &mut [u8], x: u32, y: u32, w: u32, h: u32, r: u8, g: u8, b: u8, a: u8) {
    if w == 0 || h == 0 {
        return;
    }
    let x2 = (x + w - 1).min(WIDTH - 1);
    let y2 = (y + h - 1).min(HEIGHT - 1);
    for px in x..=x2 {
        blend_pixel(frame, px, y, r, g, b, a);
        blend_pixel(frame, px, y2, r, g, b, a);
    }
    for py in y..=y2 {
        blend_pixel(frame, x, py, r, g, b, a);
        blend_pixel(frame, x2, py, r, g, b, a);
    }
}

/// Fill a single grid cell with an opaque RGB color.
fn fill_cell_rgb(frame: &mut [u8], grid_x: u32, grid_y: u32, r: u8, g: u8, b: u8) {
    let x = grid_x * GRID_SIZE;
    let y = grid_y * GRID_SIZE;
    fill_rect_rgba(frame, x, y, GRID_SIZE, GRID_SIZE, r, g, b, 255);
}

/// Fill a single grid cell with an RGBA color.
fn fill_cell_rgba(frame: &mut [u8], grid_x: u32, grid_y: u32, r: u8, g: u8, b: u8, a: u8) {
    let x = grid_x * GRID_SIZE;
    let y = grid_y * GRID_SIZE;
    fill_rect_rgba(frame, x, y, GRID_SIZE, GRID_SIZE, r, g, b, a);
}

/// Draw the game semi-transparently, tinting the snake by `color` (used to show many agents).
fn draw_game_transparent(frame: &mut [u8], game: &Game, alpha: u8, color: (u8, u8, u8)) {
    if !game.alive {
        return;
    }

    // Draw apple semi-transparent
    fill_cell_rgba(
        frame,
        game.apple.x as u32,
        game.apple.y as u32,
        220,
        50,
        50,
        alpha,
    );

    let (base_r, base_g, base_b) = color;

    // Draw snake with agent's color
    for (i, &pos) in game.snake.iter().enumerate() {
        if i == 0 {
            // Head - brighter version of agent's color
            let bright_r = (base_r as u16 * 130 / 100).min(255) as u8;
            let bright_g = (base_g as u16 * 130 / 100).min(255) as u8;
            let bright_b = (base_b as u16 * 130 / 100).min(255) as u8;
            fill_cell_rgba(
                frame,
                pos.x as u32,
                pos.y as u32,
                bright_r,
                bright_g,
                bright_b,
                alpha,
            );
        } else {
            // Body with gradient - darker with distance from head
            let fade = 1.0 - (i as f32 * 0.015).min(0.5);
            let body_r = (base_r as f32 * fade) as u8;
            let body_g = (base_g as f32 * fade) as u8;
            let body_b = (base_b as f32 * fade) as u8;
            fill_cell_rgba(
                frame,
                pos.x as u32,
                pos.y as u32,
                body_r,
                body_g,
                body_b,
                alpha,
            );
        }
    }
}

/// Draw a simple UI button with a text label.
fn draw_button(frame: &mut [u8], x: u32, y: u32, w: u32, h: u32, label: &str) {
    fill_rect_rgba(frame, x, y, w, h, 40, 40, 60, 160);
    stroke_rect_rgba(frame, x, y, w, h, 200, 200, 220, 120);
    draw_text(
        frame,
        label,
        x + 10,
        y + (h / 2 - 6),
        2,
        (230, 240, 255, 255),
    );
}

/// Check whether a point lies within a rectangle.
fn point_in_rect(px: u32, py: u32, x: u32, y: u32, w: u32, h: u32) -> bool {
    px >= x && py >= y && px < x + w && py < y + h
}

/// Returns a 5x7 bitmap glyph for a limited set of characters (ASCII-like UI font).
fn glyph_5x7(ch: char) -> Option<[u8; 7]> {
    let c = ch.to_ascii_uppercase();
    Some(match c {
        'A' => [
            0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'B' => [
            0b11110, 0b10001, 0b11110, 0b10001, 0b10001, 0b10001, 0b11110,
        ],
        'C' => [
            0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110,
        ],
        'D' => [
            0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100,
        ],
        'E' => [
            0b11111, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000, 0b11111,
        ],
        'F' => [
            0b11111, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000, 0b10000,
        ],
        'G' => [
            0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110,
        ],
        'H' => [
            0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001, 0b10001,
        ],
        'I' => [
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b11111,
        ],
        'J' => [
            0b00111, 0b00010, 0b00010, 0b00010, 0b10010, 0b10010, 0b01100,
        ],
        'K' => [
            0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001,
        ],
        'L' => [
            0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111,
        ],
        'M' => [
            0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001,
        ],
        'N' => [
            0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001,
        ],
        'O' => [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        'P' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
        'Q' => [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101,
        ],
        'R' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001,
        ],
        'S' => [
            0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
        'T' => [
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        'U' => [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        'V' => [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100,
        ],
        'W' => [
            0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001,
        ],
        'X' => [
            0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001,
        ],
        'Y' => [
            0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        'Z' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111,
        ],
        '0' => [
            0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
        ],
        '1' => [
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        '2' => [
            0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
        ],
        '3' => [
            0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
        '4' => [
            0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
        ],
        '5' => [
            0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
        ],
        '6' => [
            0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
        ],
        '7' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
        ],
        '8' => [
            0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
        ],
        '9' => [
            0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100,
        ],
        ':' => [
            0b00000, 0b00100, 0b00000, 0b00000, 0b00100, 0b00000, 0b00000,
        ],
        '+' => [
            0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000,
        ],
        '-' => [
            0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000,
        ],
        ' ' => [
            0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000,
        ],
        _ => return None,
    })
}

/// Draw a single bitmap character and return its advance in pixels.
fn draw_char(frame: &mut [u8], ch: char, x: u32, y: u32, scale: u32, col: (u8, u8, u8, u8)) -> u32 {
    if let Some(rows) = glyph_5x7(ch) {
        for (ry, row) in rows.iter().enumerate() {
            for rx in 0..5 {
                if (row >> (4 - rx)) & 1 == 1 {
                    for sy in 0..scale {
                        for sx in 0..scale {
                            blend_pixel(
                                frame,
                                x + rx as u32 * scale + sx,
                                y + ry as u32 * scale + sy,
                                col.0,
                                col.1,
                                col.2,
                                col.3,
                            );
                        }
                    }
                }
            }
        }
        5 * scale + scale
    } else {
        5 * scale + scale
    }
}

/// Draw a text string using the 5x7 glyph font.
fn draw_text(frame: &mut [u8], text: &str, x: u32, y: u32, scale: u32, col: (u8, u8, u8, u8)) {
    let mut cx = x;
    for ch in text.chars() {
        cx += draw_char(frame, ch, cx, y, scale, col);
    }
}

/// Draw a simple bar chart of best scores per epoch.
fn draw_chart(frame: &mut [u8], x: u32, y: u32, w: u32, h: u32, data: &[usize]) {
    stroke_rect_rgba(frame, x, y, w, h, 200, 200, 200, 120);
    if data.is_empty() {
        return;
    }
    let max_val = *data.iter().max().unwrap_or(&1) as u32;
    if max_val == 0 {
        return;
    }
    let bars = data.len().min(w as usize / 6);
    let bar_w = (w / bars as u32).max(2);
    for i in 0..bars {
        let v = data[data.len() - bars + i] as u32;
        let bh = (v * (h - 2)) / max_val;
        let bx = x + 1 + i as u32 * bar_w;
        let by = y + h - 1 - bh;
        fill_rect_rgba(frame, bx, by, bar_w - 1, bh, 120, 180, 255, 160);
    }
}

// ============================
// Tests
// ============================
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_dir_rotation() {
        assert_eq!(left_dir(Dir::Up), Dir::Left);
        assert_eq!(right_dir(Dir::Up), Dir::Right);
        assert_eq!(left_dir(Dir::Right), Dir::Up);
        assert_eq!(right_dir(Dir::Left), Dir::Up);
    }

    #[test]
    fn test_dir_after_action() {
        assert_eq!(dir_after_action(Dir::Up, 0), Dir::Left);
        assert_eq!(dir_after_action(Dir::Up, 1), Dir::Up);
        assert_eq!(dir_after_action(Dir::Up, 2), Dir::Right);
    }

    #[test]
    fn test_wrap_on_wall() {
        let mut g = Game::new();
        // Place head at left edge and move left: should wrap to rightmost column
        g.snake.clear();
        g.snake_set.clear();
        g.snake.push_front(Pos::new(0, 5));
        g.snake_set.insert(Pos::new(0, 5));
        g.dir = Dir::Left;
        g.update();
        assert!(g.alive);
        let head = g.snake.front().unwrap();
        assert_eq!(head.x, GRID_WIDTH as i32 - 1);
        assert_eq!(head.y, 5);
    }

    #[test]
    fn test_self_collision_death_cause() {
        let mut g = Game::new();
        // Create a body segment in front of the head to force self-collision
        g.snake.clear();
        g.snake_set.clear();
        g.snake.push_front(Pos::new(2, 2)); // head
        g.snake.push_back(Pos::new(3, 2)); // body directly to the right
        g.snake_set.insert(Pos::new(2, 2));
        g.snake_set.insert(Pos::new(3, 2));
        g.dir = Dir::Right; // move into (3,2)
        g.update();
        assert!(!g.alive);
        assert_eq!(g.last_death, DeathCause::SelfCollision);
    }

    #[test]
    fn test_evo_reproduce_keeps_population_size() {
        let mut evo = EvoTrainer::new(24);
        // Ensure there is a champion by setting a non-zero best score
        evo.scores[0] = 1;
        let mut rng = SmallRng::from_entropy();
        let mut p: PathBuf = std::env::temp_dir();
        p.push("snake_agent_test.json");
        let save_path = p.to_string_lossy().to_string();
        evo.reproduce(&mut rng, &save_path);
        assert_eq!(evo.pop.len(), evo.pop_size);
        assert_eq!(evo.games.len(), evo.pop_size);
        assert_eq!(evo.scores.len(), evo.pop_size);
    }
}
