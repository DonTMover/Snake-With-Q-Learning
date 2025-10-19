# API и контракты

Ниже кратко описаны входы/выходы и инварианты ключевых функций, чтобы упростить дальнейшую разработку.

## Game
- `fn new() -> Self` / `fn new_with_wrap(wrap: bool) -> Self`
  - Создают новую игру. Змейка из 3 сегментов по центру, случайное яблоко.
- `fn update(&mut self)`
  - Выполняет один шаг симуляции. Если `!alive` или `paused` — ничего не делает.
  - При `wrap_world=false` выход за границы завершает игру с `DeathCause::Wall`.
  - Самопересечение завершает игру с `DeathCause::SelfCollision`.
- `fn change_dir(&mut self, new_dir: Dir)`
  - Меняет направление, запрещает разворот на 180°.
- `fn draw(&self, frame: &mut [u8])` (при отключённой фиче `gpu-render`)
  - Рисует фон, яблоко, змею и часть оверлея состояний.

## QAgent
- `fn select_action<R: Rng + ?Sized>(&mut self, s: u32, rng: &mut R) -> usize`
  - Возвращает 0/1/2 (влево/прямо/вправо). С вероятностью ε — случайное действие.
- `fn learn(&mut self, s: u32, a: usize, r: f32, ns: u32, done: bool)`
  - Обновляет `Q[s,a]` по TD‑цели `r + γ max_a' Q(ns, a')` (если `!done`).
- `fn boost_exploration(&mut self)`
  - Временной подъём `epsilon` и `alpha` для рестартов эволюции.

## EvoTrainer
- `fn new(pop_size: usize) -> Self`
  - Создаёт популяцию агентов и соответствующие игры.
- `fn reset_epoch(&mut self)`
  - Обнуляет счётчики и перезапускает игры.
- `fn set_wrap_world(&mut self, wrap: bool)`
  - Переключает режим стен/обёртки и перезапускает игры (используется DQN‑режимом).
- `fn reproduce<R: Rng + ?Sized>(&mut self, rng: &mut R) -> bool`
  - Выполняет отбор/мутации/рестарты; возвращает признак появления нового глобального чемпиона.

## Утилиты
- `fn state_key(game: &Game) -> u32`
  - Кодирует состояние в 20‑битное целое. Используется как ключ Q‑таблицы и как вход DQN/NPU (после модуляции по словарю).
- `fn dir_after_action(d: Dir, a: usize) -> Dir`
  - Применяет действие к направлению (0/1/2 => влево/прямо/вправо).
- `fn mutate_qagent(agent: &mut QAgent, rng: &mut R, sigma: f32)`
  - Ставит шум на веса Q и затухает `epsilon`. Используется при эволюции/рестартах.

## DQN (`dqn.rs`)
- `struct DqnAgent` с методами:
  - `fn new(input_vocab: usize, hidden: usize, device: &Device) -> Result<Self>`
  - `fn select_action(&self, state: u32) -> Result<usize>`
  - `fn push_transition(&mut self, s: u32, a: usize, r: f32, ns: u32, done: bool)`
  - `fn train_step(&mut self, batch: usize) -> Result<()>`
  - `fn save_safetensors(&self, path: &str) -> Result<()>`
  - `fn load_safetensors(&mut self, path: &str) -> Result<()>`

## NPU (`npu.rs`)
- `struct NpuPolicy`:
  - `fn load(model_path: &str, input_vocab: usize, actions: usize) -> Result<Self>`
  - `fn select_action(&mut self, state: u32) -> Result<usize>`
