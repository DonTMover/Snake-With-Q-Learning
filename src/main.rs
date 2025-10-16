use pixels::{Error, Pixels, SurfaceTexture};
use rand::Rng;
use std::collections::{VecDeque, HashMap};
use std::time::{Duration, Instant};
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

#[derive(Clone, Copy, PartialEq, Eq)]
struct Pos {
    x: i32,
    y: i32,
}

impl Pos {
    fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Dir {
    Up,
    Down,
    Left,
    Right,
}


struct Game {
    snake: VecDeque<Pos>,
    dir: Dir,
    apple: Pos,
    alive: bool,
    score: usize,
    paused: bool,
}

impl Game {
    fn new() -> Self {
        let start_x = (GRID_WIDTH / 2) as i32;
        let start_y = (GRID_HEIGHT / 2) as i32;
        let mut snake = VecDeque::new();
        snake.push_back(Pos::new(start_x, start_y));
        snake.push_back(Pos::new(start_x - 1, start_y));
        snake.push_back(Pos::new(start_x - 2, start_y));

        let mut game = Self {
            snake,
            dir: Dir::Right,
            apple: Pos::new(0, 0),
            alive: true,
            score: 0,
            paused: false,
        };
        game.place_apple();
        game
    }

    fn place_apple(&mut self) {
        let mut rng = rand::thread_rng();
        loop {
            let x = rng.gen_range(0..GRID_WIDTH as i32);
            let y = rng.gen_range(0..GRID_HEIGHT as i32);
            let p = Pos::new(x, y);
            if !self.snake.iter().any(|&s| s == p) {
                self.apple = p;
                break;
            }
        }
    }

    fn update(&mut self) {
        if !self.alive || self.paused {
            return;
        }

        let head = self.snake.front().unwrap();
        let new_head = match self.dir {
            Dir::Up => Pos::new(head.x, head.y - 1),
            Dir::Down => Pos::new(head.x, head.y + 1),
            Dir::Left => Pos::new(head.x - 1, head.y),
            Dir::Right => Pos::new(head.x + 1, head.y),
        };

        // Check collision with walls
        if new_head.x < 0
            || new_head.x >= GRID_WIDTH as i32
            || new_head.y < 0
            || new_head.y >= GRID_HEIGHT as i32
        {
            self.alive = false;
            return;
        }

        // Check collision with self
        if self.snake.iter().any(|&s| s == new_head) {
            self.alive = false;
            return;
        }

        self.snake.push_front(new_head);

        // Check if apple eaten
        if new_head == self.apple {
            self.score += 1;
            self.place_apple();
        } else {
            self.snake.pop_back();
        }
    }

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

    fn draw(&self, frame: &mut [u8]) {
        // Clear screen with dark background
        clear_rgba(frame, 20, 20, 30, 255);

        // Draw grid
        for y in 0..GRID_HEIGHT {
            for x in 0..GRID_WIDTH {
                if (x + y) % 2 == 0 {
                    self.draw_rect(frame, x, y, 25, 25, 35);
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
            draw_text(frame, "GAME OVER", WIDTH / 2 - 80, HEIGHT / 2 - 20, 2, (255, 100, 100, 255));
            draw_text(frame, &format!("SCORE: {}", self.score), WIDTH / 2 - 70, HEIGHT / 2 + 20, 2, (255, 255, 255, 255));
            draw_text(frame, "PRESS R TO RESTART", WIDTH / 2 - 130, HEIGHT / 2 + 60, 2, (200, 200, 200, 255));
        } else if self.paused {
            draw_text(frame, "PAUSED", WIDTH / 2 - 50, HEIGHT / 2, 2, (255, 255, 100, 255));
        }

        // Note: Score/Length are drawn inside the overlay panel (same plane) in the RedrawRequested block
    }

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
// Simple Q-learning Agent
// ============================

#[derive(Clone)]
struct QAgent {
    q: HashMap<u16, [f32; 3]>,
    epsilon: f32,
    min_epsilon: f32,
    decay: f32,
    alpha: f32,
    gamma: f32,
    steps: u64,
    episodes: u64,
}

impl QAgent {
    fn new() -> Self {
        Self { q: HashMap::new(), epsilon: 0.3, min_epsilon: 0.1, decay: 0.9985, alpha: 0.4, gamma: 0.95, steps: 0, episodes: 0 }
    }

    fn get_qs(&mut self, s: u16) -> &mut [f32;3] { self.q.entry(s).or_insert([0.0, 0.0, 0.0]) }

    fn select_action(&mut self, s: u16, rng: &mut rand::rngs::ThreadRng) -> usize {
        if rng.r#gen::<f32>() < self.epsilon { rng.gen_range(0..3) } else {
            let qs = *self.get_qs(s);
            if qs[0] >= qs[1] && qs[0] >= qs[2] { 0 } else if qs[1] >= qs[2] { 1 } else { 2 }
        }
    }

    fn learn(&mut self, s: u16, a: usize, r: f32, ns: u16, done: bool) {
        let next_max = if done {
            0.0
        } else {
            let nqs = self.q.get(&ns).copied().unwrap_or([0.0;3]);
            nqs[0].max(nqs[1]).max(nqs[2])
        };
        let alpha = self.alpha; let gamma = self.gamma;
        let qsa = self.get_qs(s);
        let td_target = r + gamma * next_max;
        qsa[a] = qsa[a] + alpha * (td_target - qsa[a]);
    }
}

// ============================
// Evolutionary trainer (population of agents)
// ============================

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
    games: Vec<Game>, // parallel games for each individual
}

impl EvoTrainer {
    fn new(pop_size: usize) -> Self {
        let mut pop = Vec::with_capacity(pop_size);
        let mut games = Vec::with_capacity(pop_size);
        for _ in 0..pop_size { 
            pop.push(QAgent::new());
            games.push(Game::new());
        }
        let max_apples = (GRID_WIDTH as usize * GRID_HEIGHT as usize).saturating_sub(3); // 3 is initial snake length
        Self { training: false, solved: false, pop, pop_size, current: 0, epoch: 0, epoch_best: Vec::new(), scores: vec![0; pop_size], step_limit: 3000, steps_taken: 0, target_score: max_apples, best_score: 0, games }
    }

    fn reset_epoch(&mut self) { 
        self.current = 0; 
        self.steps_taken = 0; 
        self.scores.fill(0);
        for i in 0..self.pop_size {
            self.games[i] = Game::new();
        }
    }

    fn reproduce(&mut self, rng: &mut rand::rngs::ThreadRng) {
        let mut idxs: Vec<usize> = (0..self.pop_size).collect();
        idxs.sort_by_key(|&i| std::cmp::Reverse(self.scores[i]));
        let best_idx = *idxs.first().unwrap_or(&0);
        let best_score = self.scores[best_idx];
        self.epoch_best.push(best_score);
        self.best_score = self.best_score.max(best_score);

        // Keep the single best parent; fill rest with its mutated copies
        let parent = self.pop[best_idx].clone();
        let mut new_pop: Vec<QAgent> = Vec::with_capacity(self.pop_size);
        new_pop.push(parent.clone());
        while new_pop.len() < self.pop_size {
            let mut child = parent.clone();
            mutate_qagent(&mut child, rng, 0.25);
            new_pop.push(child);
        }
        self.pop = new_pop;
        self.epoch += 1;
        self.reset_epoch();
    }
}

fn mutate_qagent(agent: &mut QAgent, rng: &mut rand::rngs::ThreadRng, sigma: f32) {
    for arr in agent.q.values_mut() {
        for v in arr.iter_mut() { *v += rng.gen_range(-sigma..sigma); }
    }
    agent.epsilon = (agent.epsilon * agent.decay).max(agent.min_epsilon);
}

fn left_dir(d: Dir) -> Dir { match d { Dir::Up=>Dir::Left, Dir::Left=>Dir::Down, Dir::Down=>Dir::Right, Dir::Right=>Dir::Up } }
fn right_dir(d: Dir) -> Dir { match d { Dir::Up=>Dir::Right, Dir::Right=>Dir::Down, Dir::Down=>Dir::Left, Dir::Left=>Dir::Up } }
fn dir_after_action(d: Dir, a: usize) -> Dir { match a { 0=>left_dir(d), 1=>d, _=>right_dir(d) } }

fn offset_for_dir(d: Dir) -> (i32,i32) { match d { Dir::Up=>(0,-1), Dir::Down=>(0,1), Dir::Left=>(-1,0), Dir::Right=>(1,0) } }

fn will_collide(game: &Game, dir: Dir) -> bool {
    let head = game.snake.front().unwrap();
    let (dx,dy) = offset_for_dir(dir);
    let nx = head.x + dx; let ny = head.y + dy;
    if nx < 0 || ny < 0 || nx >= GRID_WIDTH as i32 || ny >= GRID_HEIGHT as i32 { return true; }
    let np = Pos::new(nx, ny);
    game.snake.iter().any(|&s| s == np)
}

fn apple_relative_flags(game: &Game) -> (bool,bool,bool) {
    let head = game.snake.front().unwrap();
    let (dx,dy) = (game.apple.x - head.x, game.apple.y - head.y);
    match game.dir {
        Dir::Up => (dx < 0, dy < 0, dx > 0),
        Dir::Down => (dx > 0, dy > 0, dx < 0),
        Dir::Left => (dy > 0, dx < 0, dy < 0),
        Dir::Right => (dy < 0, dx > 0, dy > 0),
    }
}

fn state_key(game: &Game) -> u16 {
    // Bits: 0 danger_left, 1 danger_ahead, 2 danger_right, 3 apple_left, 4 apple_straight, 5 apple_right
    let dl = will_collide(game, left_dir(game.dir));
    let da = will_collide(game, game.dir);
    let dr = will_collide(game, right_dir(game.dir));
    let (al, as_, ar) = apple_relative_flags(game);
    let mut k: u16 = 0;
    if dl { k |= 1<<0; } if da { k |= 1<<1; } if dr { k |= 1<<2; }
    if al { k |= 1<<3; }
    if as_ { k |= 1<<4; }
    if ar { k |= 1<<5; }
    k
}

fn main() -> Result<(), Error> {
    let event_loop = EventLoop::new();
    let mut input = WinitInputHelper::new();

    let window = WindowBuilder::new()
        .with_title("🐍 Snake Game")
        .with_inner_size(LogicalSize::new(WIDTH, HEIGHT))
        .with_resizable(false)
        .build(&event_loop)
        .unwrap();

    let mut pixels = {
        let window_size = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        Pixels::new(WIDTH, HEIGHT, surface_texture)?
    };

    let mut game = Game::new();
    let mut evo = EvoTrainer::new(10);
    let mut rng = rand::thread_rng();
    let mut last_update = Instant::now();
    let mut tick_duration = Duration::from_millis(150);
    let mut manual_speed_delta_ms: i32 = 0;
    let mut evo_steps_per_frame: u32 = 1200; // evolution training speed (steps processed per frame)

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        if let Event::RedrawRequested(_) = event {
            let frame = pixels.frame_mut();
            
            // Draw the appropriate game(s)
            if evo.training {
                // Draw all 10 individuals semi-transparently
                clear_rgba(frame, 20, 20, 30, 255);
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
                                            frame[idx] = 25; frame[idx + 1] = 25; frame[idx + 2] = 35; frame[idx + 3] = 255;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                
                // Draw all individuals
                for g in &evo.games {
                    draw_game_transparent(frame, g, 80); // alpha 80 for semi-transparent
                }
            } else {
                game.draw(frame);
            }

            // Controls overlay (semi-transparent)
            let panel_x: u32 = 8;
            let panel_y: u32 = 8;
            let panel_w: u32 = 280;
            let panel_h: u32 = 460; // increased to fit EVO status and chart
            let btn_h: u32 = 28;
            let btn_w: u32 = panel_w - 16;
            let btn_x: u32 = panel_x + 8;
            // Chart area inside panel
            let chart_y: u32 = panel_y + 246;
            let chart_h: u32 = 80;
            let btn1_y: u32 = chart_y + chart_h + 8; // start buttons after chart
            let btn2_y: u32 = btn1_y + btn_h + 6;
            let btn3_y: u32 = btn2_y + btn_h + 6;

            fill_rect_rgba(frame, panel_x, panel_y, panel_w, panel_h, 0, 0, 0, 140);
            stroke_rect_rgba(frame, panel_x, panel_y, panel_w, panel_h, 255, 255, 255, 60);
            draw_text(frame, "CONTROLS", panel_x + 10, panel_y + 10, 2, (180, 220, 255, 255));
            // HUD inside panel with extra line spacing
            draw_text(frame, &format!("SCORE: {}", game.score), panel_x + 10, panel_y + 34, 2, (230, 230, 230, 255));
            draw_text(frame, &format!("LENGTH: {}", game.snake.len()), panel_x + 10, panel_y + 60, 2, (200, 200, 200, 255));
            // Speed indicator (based on tick duration)
            let ms = tick_duration.as_millis() as f32;
            let sps = if ms > 0.0 { 1000.0 / ms } else { 0.0 };
            draw_text(frame, &format!("SPEED: {} ms (~{:.1}/s)", ms as i32, sps), panel_x + 10, panel_y + 90, 2, (200, 220, 255, 255));

            // Evolutionary training status
            draw_text(frame, &format!("EVO: {} (E)", if evo.training {"ON"} else {"OFF"}), panel_x + 10, panel_y + 116, 2, (220, 200, 240, 255));
            let alive_count = evo.games.iter().filter(|g| g.alive).count();
            draw_text(frame, &format!("EPOCH: {}  ALIVE: {}/{}", evo.epoch, alive_count, evo.pop_size), panel_x + 10, panel_y + 142, 2, (220, 200, 240, 255));
            draw_text(frame, &format!("TARGET: {}  BEST: {}", evo.target_score, evo.best_score), panel_x + 10, panel_y + 168, 2, (220, 200, 240, 255));
            draw_text(frame, &format!("EVO SPD: {} steps/frame (+/-)", evo_steps_per_frame), panel_x + 10, panel_y + 194, 2, (200, 220, 255, 255));
            // Chart of best apples per epoch
            draw_chart(frame, panel_x + 10, chart_y, panel_w - 20, chart_h, &evo.epoch_best);

            let paused_label = if game.paused { "RESUME  P" } else { "PAUSE   P" };
            draw_button(frame, btn_x, btn1_y, btn_w, btn_h, paused_label);
            draw_button(frame, btn_x, btn2_y, btn_w, btn_h, "SPEED+  +");
            draw_button(frame, btn_x, btn3_y, btn_w, btn_h, "RESTART R");

            if pixels.render().is_err() {
                *control_flow = ControlFlow::Exit;
            }
        }

        if input.update(&event) {
            // Handle quit
            if input.key_pressed(VirtualKeyCode::Escape) || input.close_requested() || input.destroyed() {
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
                    evo.solved = false; evo.reset_epoch(); evo.epoch = 0; evo.epoch_best.clear(); evo.best_score = 0; game = Game::new();
                }
            }

            // Speed controls (keyboard)
            if evo.training {
                if input.key_pressed(VirtualKeyCode::NumpadAdd) || input.key_pressed(VirtualKeyCode::Equals) {
                    evo_steps_per_frame = (evo_steps_per_frame.saturating_mul(2)).min(10_000);
                }
                if input.key_pressed(VirtualKeyCode::NumpadSubtract) || input.key_pressed(VirtualKeyCode::Minus) {
                    evo_steps_per_frame = (evo_steps_per_frame / 2).max(1);
                }
            } else {
                if input.key_pressed(VirtualKeyCode::NumpadAdd) || input.key_pressed(VirtualKeyCode::Equals) { manual_speed_delta_ms = (manual_speed_delta_ms - 10).max(-150); }
                if input.key_pressed(VirtualKeyCode::NumpadSubtract) || input.key_pressed(VirtualKeyCode::Minus) { manual_speed_delta_ms = (manual_speed_delta_ms + 10).min(300); }
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
            if let Some((mx, my)) = input.mouse() {
                if input.mouse_pressed(0) {
                    let mx = mx as u32; let my = my as u32;
                    let panel_x: u32 = 8; let panel_y: u32 = 8; let panel_w: u32 = 280; let btn_h: u32 = 28; let btn_w: u32 = panel_w - 16; let btn_x: u32 = panel_x + 8; let chart_y: u32 = panel_y + 246; let chart_h: u32 = 80; let btn1_y: u32 = chart_y + chart_h + 8; let btn2_y: u32 = btn1_y + btn_h + 6; let btn3_y: u32 = btn2_y + btn_h + 6;
                    if point_in_rect(mx, my, btn_x, btn1_y, btn_w, btn_h) { game.paused = !game.paused; }
                    else if point_in_rect(mx, my, btn_x, btn2_y, btn_w, btn_h) {
                        if evo.training { evo_steps_per_frame = (evo_steps_per_frame.saturating_mul(2)).min(10_000); }
                        else { manual_speed_delta_ms = (manual_speed_delta_ms - 10).max(-150); }
                    }
                    else if point_in_rect(mx, my, btn_x, btn3_y, btn_w, btn_h) { game = Game::new(); tick_duration = Duration::from_millis(150); }
                }
            }

            // Evolutionary training loop (population of agents - all run in parallel)
            if evo.training {
                let steps_per_frame: u32 = evo_steps_per_frame.max(1);
                if game.paused { window.request_redraw(); return; }
                
                for _ in 0..steps_per_frame {
                    let mut all_done = true;
                    
                    // Update all individuals in parallel
                    for i in 0..evo.pop_size {
                        let g = &mut evo.games[i];
                        if !g.alive || evo.scores[i] >= evo.target_score {
                            continue; // skip finished individuals
                        }
                        all_done = false;
                        
                        let agent = &mut evo.pop[i];
                        let s = state_key(g);
                        let a_idx = agent.select_action(s, &mut rng);
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
                        let mut reward = if died { -10.0 } else if ate { 10.0 } else { -0.01 };
                        if !died && !ate {
                            if d1 < d0 { reward += 0.02; } else if d1 > d0 { reward -= 0.02; }
                        }
                        let ns = state_key(g);
                        agent.learn(s, a_idx, reward, ns, died || !g.alive);
                        agent.steps += 1;
                        if died { 
                            agent.episodes += 1; 
                            agent.epsilon = (agent.epsilon * agent.decay).max(agent.min_epsilon);
                        }
                        if g.alive {
                            evo.scores[i] = g.score;
                        }
                        if g.score >= evo.target_score {
                            evo.solved = true;
                            evo.training = false;
                        }
                    }
                    
                    evo.steps_taken += 1;
                    if all_done || evo.steps_taken >= evo.step_limit {
                        // All individuals finished or step limit reached - start new epoch
                        evo.reproduce(&mut rng);
                        break;
                    }
                }
                window.request_redraw();
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

fn clear_rgba(frame: &mut [u8], r: u8, g: u8, b: u8, a: u8) {
    for px in frame.chunks_exact_mut(4) { px[0]=r; px[1]=g; px[2]=b; px[3]=a; }
}

fn blend_pixel(frame: &mut [u8], x: u32, y: u32, r: u8, g: u8, b: u8, a: u8) {
    if x>=WIDTH || y>=HEIGHT {return;} let idx=((y*WIDTH+x)*4) as usize; if idx+3>=frame.len(){return;}
    let ar=a as u16; let iar=(255-a) as u16; let dr=frame[idx] as u16; let dg=frame[idx+1] as u16; let db=frame[idx+2] as u16;
    frame[idx]   = (((r as u16)*ar + dr*iar)/255) as u8;
    frame[idx+1] = (((g as u16)*ar + dg*iar)/255) as u8;
    frame[idx+2] = (((b as u16)*ar + db*iar)/255) as u8;
    frame[idx+3] = 255;
}

fn fill_rect_rgba(frame: &mut [u8], x: u32, y: u32, w: u32, h: u32, r: u8, g: u8, b: u8, a: u8) {
    let x2=(x+w).min(WIDTH); let y2=(y+h).min(HEIGHT);
    for py in y..y2 { for px in x..x2 { blend_pixel(frame, px, py, r,g,b,a); } }
}

fn stroke_rect_rgba(frame: &mut [u8], x: u32, y: u32, w: u32, h: u32, r: u8, g: u8, b: u8, a: u8) {
    if w==0||h==0 {return;} let x2=(x+w-1).min(WIDTH-1); let y2=(y+h-1).min(HEIGHT-1);
    for px in x..=x2 { blend_pixel(frame, px, y, r,g,b,a); blend_pixel(frame, px, y2, r,g,b,a);} 
    for py in y..=y2 { blend_pixel(frame, x, py, r,g,b,a); blend_pixel(frame, x2, py, r,g,b,a);} 
}

fn fill_cell_rgb(frame: &mut [u8], grid_x: u32, grid_y: u32, r: u8, g: u8, b: u8) {
    let x=grid_x*GRID_SIZE; let y=grid_y*GRID_SIZE; fill_rect_rgba(frame, x, y, GRID_SIZE, GRID_SIZE, r,g,b,255);
}

fn fill_cell_rgba(frame: &mut [u8], grid_x: u32, grid_y: u32, r: u8, g: u8, b: u8, a: u8) {
    let x=grid_x*GRID_SIZE; let y=grid_y*GRID_SIZE; fill_rect_rgba(frame, x, y, GRID_SIZE, GRID_SIZE, r,g,b,a);
}

fn draw_game_transparent(frame: &mut [u8], game: &Game, alpha: u8) {
    if !game.alive { return; }
    
    // Draw apple semi-transparent
    fill_cell_rgba(frame, game.apple.x as u32, game.apple.y as u32, 220, 50, 50, alpha);
    
    // Draw snake semi-transparent
    for (i, &pos) in game.snake.iter().enumerate() {
        if i == 0 {
            // Head
            fill_cell_rgba(frame, pos.x as u32, pos.y as u32, 100, 255, 100, alpha);
        } else {
            // Body with gradient
            let brightness = 200 - (i * 10).min(100) as u8;
            fill_cell_rgba(frame, pos.x as u32, pos.y as u32, 50, brightness, 50, alpha);
        }
    }
}

fn draw_button(frame: &mut [u8], x: u32, y: u32, w: u32, h: u32, label: &str) {
    fill_rect_rgba(frame, x, y, w, h, 40, 40, 60, 160);
    stroke_rect_rgba(frame, x, y, w, h, 200, 200, 220, 120);
    draw_text(frame, label, x+10, y + (h/2 - 6), 2, (230,240,255,255));
}

fn point_in_rect(px: u32, py: u32, x: u32, y: u32, w: u32, h: u32) -> bool { px>=x && py>=y && px<x+w && py<y+h }

fn glyph_5x7(ch: char) -> Option<[u8;7]> {
    let c=ch.to_ascii_uppercase();
    Some(match c {
        'A'=>[0b01110,0b10001,0b10001,0b11111,0b10001,0b10001,0b10001],
        'B'=>[0b11110,0b10001,0b11110,0b10001,0b10001,0b10001,0b11110],
        'C'=>[0b01110,0b10001,0b10000,0b10000,0b10000,0b10001,0b01110],
        'D'=>[0b11100,0b10010,0b10001,0b10001,0b10001,0b10010,0b11100],
        'E'=>[0b11111,0b10000,0b11110,0b10000,0b10000,0b10000,0b11111],
        'F'=>[0b11111,0b10000,0b11110,0b10000,0b10000,0b10000,0b10000],
        'G'=>[0b01110,0b10001,0b10000,0b10111,0b10001,0b10001,0b01110],
        'H'=>[0b10001,0b10001,0b11111,0b10001,0b10001,0b10001,0b10001],
        'I'=>[0b11111,0b00100,0b00100,0b00100,0b00100,0b00100,0b11111],
        'J'=>[0b00111,0b00010,0b00010,0b00010,0b10010,0b10010,0b01100],
        'K'=>[0b10001,0b10010,0b10100,0b11000,0b10100,0b10010,0b10001],
        'L'=>[0b10000,0b10000,0b10000,0b10000,0b10000,0b10000,0b11111],
        'M'=>[0b10001,0b11011,0b10101,0b10101,0b10001,0b10001,0b10001],
        'N'=>[0b10001,0b11001,0b10101,0b10011,0b10001,0b10001,0b10001],
        'O'=>[0b01110,0b10001,0b10001,0b10001,0b10001,0b10001,0b01110],
        'P'=>[0b11110,0b10001,0b10001,0b11110,0b10000,0b10000,0b10000],
        'Q'=>[0b01110,0b10001,0b10001,0b10001,0b10101,0b10010,0b01101],
        'R'=>[0b11110,0b10001,0b10001,0b11110,0b10100,0b10010,0b10001],
        'S'=>[0b01111,0b10000,0b10000,0b01110,0b00001,0b00001,0b11110],
        'T'=>[0b11111,0b00100,0b00100,0b00100,0b00100,0b00100,0b00100],
        'U'=>[0b10001,0b10001,0b10001,0b10001,0b10001,0b10001,0b01110],
        'V'=>[0b10001,0b10001,0b10001,0b10001,0b10001,0b01010,0b00100],
        'W'=>[0b10001,0b10001,0b10001,0b10101,0b10101,0b11011,0b10001],
        'X'=>[0b10001,0b10001,0b01010,0b00100,0b01010,0b10001,0b10001],
        'Y'=>[0b10001,0b10001,0b01010,0b00100,0b00100,0b00100,0b00100],
        'Z'=>[0b11111,0b00001,0b00010,0b00100,0b01000,0b10000,0b11111],
        '0'=>[0b01110,0b10001,0b10011,0b10101,0b11001,0b10001,0b01110],
        '1'=>[0b00100,0b01100,0b00100,0b00100,0b00100,0b00100,0b01110],
        '2'=>[0b01110,0b10001,0b00001,0b00010,0b00100,0b01000,0b11111],
        '3'=>[0b11110,0b00001,0b00001,0b01110,0b00001,0b00001,0b11110],
        '4'=>[0b00010,0b00110,0b01010,0b10010,0b11111,0b00010,0b00010],
        '5'=>[0b11111,0b10000,0b11110,0b00001,0b00001,0b10001,0b01110],
        '6'=>[0b00110,0b01000,0b10000,0b11110,0b10001,0b10001,0b01110],
        '7'=>[0b11111,0b00001,0b00010,0b00100,0b01000,0b01000,0b01000],
        '8'=>[0b01110,0b10001,0b10001,0b01110,0b10001,0b10001,0b01110],
        '9'=>[0b01110,0b10001,0b10001,0b01111,0b00001,0b00010,0b01100],
        ':'=>[0b00000,0b00100,0b00000,0b00000,0b00100,0b00000,0b00000],
        '+'=>[0b00000,0b00100,0b00100,0b11111,0b00100,0b00100,0b00000],
        '-'=>[0b00000,0b00000,0b00000,0b11111,0b00000,0b00000,0b00000],
        ' '=>[0b00000,0b00000,0b00000,0b00000,0b00000,0b00000,0b00000],
        _ => return None,
    })
}

fn draw_char(frame: &mut [u8], ch: char, x: u32, y: u32, scale: u32, col: (u8,u8,u8,u8)) -> u32 {
    if let Some(rows)=glyph_5x7(ch){
        for (ry,row) in rows.iter().enumerate(){
            for rx in 0..5 { if (row >> (4-rx)) & 1 == 1 {
                for sy in 0..scale { for sx in 0..scale {
                    blend_pixel(frame, x + rx as u32*scale + sx, y + ry as u32*scale + sy, col.0,col.1,col.2,col.3);
                }}
            }}
        }
        5*scale + scale
    } else { 5*scale + scale }
}

fn draw_text(frame: &mut [u8], text: &str, x: u32, y: u32, scale: u32, col: (u8,u8,u8,u8)) {
    let mut cx=x; for ch in text.chars(){ cx += draw_char(frame, ch, cx, y, scale, col); }
}

fn draw_chart(frame: &mut [u8], x: u32, y: u32, w: u32, h: u32, data: &Vec<usize>) {
    stroke_rect_rgba(frame, x, y, w, h, 200,200,200,120);
    if data.is_empty() { return; }
    let max_val = *data.iter().max().unwrap_or(&1) as u32;
    if max_val == 0 { return; }
    let bars = data.len().min(w as usize / 6);
    let bar_w = (w / bars as u32).max(2);
    for i in 0..bars {
        let v = data[data.len()-bars + i] as u32;
        let bh = (v * (h-2)) / max_val;
        let bx = x + 1 + i as u32 * bar_w;
        let by = y + h - 1 - bh;
        fill_rect_rgba(frame, bx, by, bar_w - 1, bh, 120, 180, 255, 160);
    }
}