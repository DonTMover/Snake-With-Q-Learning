use crate::pos::Pos;
use rand::Rng;
use std::cmp::min;
use std::collections::VecDeque;

#[derive(Clone, Copy)]
pub enum Dir {
    Up,
    Down,
    Left,
    Right,
}

pub struct Game {
    pub width: u16,
    pub height: u16,
    pub snake: VecDeque<Pos>,
    pub dir: Dir,
    pub apple: Pos,
    pub alive: bool,
    pub score: usize,
    pub pause: bool,
    pub horizontal_counter: u8, // Counter for aspect ratio correction
}

impl Game {
    pub fn new(width: u16, height: u16) -> Self {
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
            horizontal_counter: 0,
        };
        g.place_apple();
        g
    }

    pub fn place_apple(&mut self) {
        let mut rng = rand::thread_rng();
        loop {
            let x = rng.gen_range(1..self.width - 1);
            let y = rng.gen_range(1..self.height - 1);
            let p = Pos::new(x, y);
            if !self.snake_contains(p) {
                self.apple = p;
                break;
            }
        }
    }

    pub fn snake_contains(&self, p: Pos) -> bool {
        self.snake.iter().any(|&s| s == p)
    }

    // Returns true if an apple was eaten on this step
    pub fn step(&mut self) -> bool {
        if !self.alive || self.pause {
            return false;
        }
        
        // Aspect ratio correction: horizontal movement should be slower
        // Move horizontally only every 2nd call
        let should_move = match self.dir {
            Dir::Left | Dir::Right => {
                self.horizontal_counter += 1;
                if self.horizontal_counter >= 2 {
                    self.horizontal_counter = 0;
                    true
                } else {
                    false
                }
            }
            Dir::Up | Dir::Down => true,
        };
        
        if !should_move {
            return false;
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

        if new_head.x == 0 || new_head.x == self.width - 1 || new_head.y == 0 || new_head.y == self.height - 1 {
            self.alive = false;
            return false;
        }

        if self.snake_contains(new_head) {
            self.alive = false;
            return false;
        }

        self.snake.push_front(new_head);
        // Apple?
        if new_head == self.apple {
            self.score += 1;
            self.place_apple();
            return true;
        } else {
            self.snake.pop_back();
        }
        false
    }

    pub fn change_dir(&mut self, new_dir: Dir) {
        match (self.dir, new_dir) {
            (Dir::Up, Dir::Down) | (Dir::Down, Dir::Up) | (Dir::Left, Dir::Right) | (Dir::Right, Dir::Left) => {}
            _ => self.dir = new_dir,
        }
    }
}
