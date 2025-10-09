use crossterm::{
    cursor::{Hide, MoveTo, Show},
    event::{poll, read, Event, KeyCode, KeyEvent, KeyModifiers},
    execute,
    style::{Color, PrintStyledContent, Stylize},
    terminal::{
        disable_raw_mode, enable_raw_mode, Clear, ClearType, EnterAlternateScreen,
        LeaveAlternateScreen, SetSize,
    },
};
use rand::Rng;
use std::cmp::min;
use std::collections::VecDeque;
use std::io::{stdout, Write};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq, Eq)]
struct Pos {
    x: u16,
    y: u16,
}

impl Pos {
    fn new(x: u16, y: u16) -> Self {
        Self { x, y }
    }
}

#[derive(Clone, Copy)]
enum Dir {
    Up,
    Down,
    Left,
    Right,
}

struct Game {
    width: u16,
    height: u16,
    snake: VecDeque<Pos>,
    dir: Dir,
    apple: Pos,
    alive: bool,
    score: usize,
    pause: bool,
}

impl Game {
    fn new(width: u16, height: u16) -> Self {
        let start_x = width / 2;
        let start_y = height / 2;
        let mut snake = VecDeque::new();
        snake.push_back(Pos::new(start_x, start_y));
        snake.push_back(Pos::new(start_x - 1, start_y));
        snake.push_back(Pos::new(start_x - 2, start_y));

        let mut g = Self {
            width,
            height,
            snake,
            dir: Dir::Right,
            apple: Pos::new(0, 0),
            alive: true,
            score: 0,
            pause: false,
        };
        g.place_apple();
        g
    }

    fn place_apple(&mut self) {
        let mut rng = rand::thread_rng();
        loop {
            // keep apple inside inner area (1..=width-2, 1..=height-2)
            let x = rng.gen_range(1..self.width - 1);
            let y = rng.gen_range(1..self.height - 1);
            let p = Pos::new(x, y);
            if !self.snake_contains(p) {
                self.apple = p;
                break;
            }
        }
    }

    fn snake_contains(&self, p: Pos) -> bool {
        self.snake.iter().any(|&s| s == p)
    }

    fn step(&mut self) {
        if !self.alive || self.pause {
            return;
        }
        let head = self.snake.front().unwrap();
        let new_head = match self.dir {
            Dir::Up => {
                if head.y == 1 {
                    Pos::new(head.x, 0)
                } else {
                    Pos::new(head.x, head.y - 1)
                }
            }
            Dir::Down => Pos::new(head.x, min(head.y + 1, self.height - 1)),
            Dir::Left => Pos::new(head.x.saturating_sub(1), head.y),
            Dir::Right => Pos::new(min(head.x + 1, self.width - 1), head.y),
        };

        // Check wall collision (if head at 0 or width-1 or height-1)
        if new_head.x == 0 || new_head.x == self.width - 1 || new_head.y == 0 || new_head.y == self.height - 1 {
            self.alive = false;
            return;
        }

        // Check self collision
        if self.snake_contains(new_head) {
            self.alive = false;
            return;
        }

        self.snake.push_front(new_head);

        // Apple?
        if new_head == self.apple {
            self.score += 1;
            self.place_apple();
        } else {
            self.snake.pop_back();
        }
    }

    fn change_dir(&mut self, new_dir: Dir) {
        // prevent reverse
        match (self.dir, new_dir) {
            (Dir::Up, Dir::Down) | (Dir::Down, Dir::Up) | (Dir::Left, Dir::Right) | (Dir::Right, Dir::Left) => {}
            _ => self.dir = new_dir,
        }
    }
}

fn draw<W: Write>(out: &mut W, game: &Game) -> crossterm::Result<()> {
    execute!(out, MoveTo(0, 0))?;
    // Draw top border
    for x in 0..game.width {
        print!("#");
    }
    println!();

    // Draw middle
    for y in 1..game.height - 1 {
        for x in 0..game.width {
            if x == 0 || x == game.width - 1 {
                print!("#");
            } else {
                let p = Pos::new(x, y);
                if p == game.apple {
                    // apple
                    print!("{}", "◉".red());
                } else if game.snake_contains(p) {
                    // snake body / head
                    let head = game.snake.front().unwrap();
                    if *head == p {
                        print!("{}", "█".green());
                    } else {
                        print!("{}", "▓".green());
                    }
                } else {
                    print!(" ");
                }
            }
        }
        println!();
    }

    // Draw bottom border
    for _ in 0..game.width {
        print!("#");
    }
    println!();

    // Score and info line
    if game.alive {
        println!("Score: {}    {}  (q=quit, p=pause, arrows/WASD to move)", game.score, if game.pause { "PAUSED" } else { "" });
    } else {
        println!("Game Over! Score: {}    (q=quit, r=restart)", game.score);
    }

    out.flush()?;
    Ok(())
}

fn restore_terminal() {
    let mut out = stdout();
    let _ = disable_raw_mode();
    let _ = execute!(out, Show, LeaveAlternateScreen);
}

fn main() -> crossterm::Result<()> {
    // Setup terminal
    let mut out = stdout();
    enable_raw_mode()?;
    execute!(out, EnterAlternateScreen, Hide)?;

    // Determine terminal size and set a comfortable game area (clamp)
    let (term_width, term_height) = crossterm::terminal::size()?;
    // Ensure minimum size
    let width = if term_width < 20 { 20 } else { term_width };
    let height = if term_height < 10 { 10 } else { term_height };

    // Optionally set terminal to desired size (best-effort)
    let _ = execute!(out, SetSize(width, height));

    let mut game = Game::new(width, height);

    let mut last_tick = Instant::now();
    let mut tick_ms: u64 = 150;

    // main loop
    loop {
        // adjust speed slightly by score (faster as score increases)
        let speed = tick_ms.saturating_sub((game.score as u64) * 3);
        let tick = Duration::from_millis(min(speed, 250));

        // handle input (non-blocking)
        if poll(Duration::from_millis(10))? {
            if let Event::Key(KeyEvent { code, modifiers: _ }) = read()? {
                match code {
                    KeyCode::Char('q') | KeyCode::Char('Q') => {
                        restore_terminal();
                        return Ok(());
                    }
                    KeyCode::Char('p') | KeyCode::Char('P') => {
                        game.pause = !game.pause;
                    }
                    KeyCode::Char('r') | KeyCode::Char('R') => {
                        if !game.alive {
                            game = Game::new(width, height);
                        }
                    }
                    KeyCode::Up => game.change_dir(Dir::Up),
                    KeyCode::Down => game.change_dir(Dir::Down),
                    KeyCode::Left => game.change_dir(Dir::Left),
                    KeyCode::Right => game.change_dir(Dir::Right),
                    KeyCode::Char('w') | KeyCode::Char('W') => game.change_dir(Dir::Up),
                    KeyCode::Char('s') | KeyCode::Char('S') => game.change_dir(Dir::Down),
                    KeyCode::Char('a') | KeyCode::Char('A') => game.change_dir(Dir::Left),
                    KeyCode::Char('d') | KeyCode::Char('D') => game.change_dir(Dir::Right),
                    KeyCode::Char('c') if cfg!(windows) && KeyModifiers::CONTROL == KeyModifiers::CONTROL => {
                        // ignore, ctrl-c will not reach here typically
                    }
                    _ => {}
                }
            }
        }

        if last_tick.elapsed() >= tick {
            if game.alive && !game.pause {
                game.step();
            }
            execute!(out, Clear(ClearType::All))?;
            draw(&mut out, &game)?;
            last_tick = Instant::now();
        }
    }
}