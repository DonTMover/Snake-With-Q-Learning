# Обзор кода (Code Map)

Этот документ перечисляет ключевые типы, функции и их назначение. Файл предназначен для быстрой навигации по проекту.

## Константы экрана/сетки
- `WIDTH: u32 = 800`, `HEIGHT: u32 = 600`, `GRID_SIZE: u32 = 20`
- `GRID_WIDTH`, `GRID_HEIGHT` — размер сетки в ячейках.

## Типы и структуры
- `Pos { x: i32, y: i32 }` — позиция в клетках.
- `Dir { Up, Down, Left, Right }` — направление движения.
- `DeathCause { None, SelfCollision, Wall }` — причина «смерти» (для shaping).
- `Game` — состояние одной игры:
  - `snake: VecDeque<Pos>` — сегменты змеи (голова в начале).
  - `snake_set: HashSet<Pos>` — для быстрых проверок самопересечения.
  - `dir: Dir`, `apple: Pos`, `alive: bool`, `score: usize`, `paused: bool`.
  - `last_death: DeathCause` — последняя причина смерти.
  - `wrap_world: bool` — обёртка по краям (true) или твёрдые стены (false).
- `QAgent` — табличный Q‑агент:
  - `q: AHashMap<u32,[f32;3]>`, параметры ε‑жадной политики и обучения.
  - Методы `new`, `select_action`, `learn`, `boost_exploration`.
- `EvoTrainer` — тренер эволюции для популяции агентов и их игр:
  - `pop`, `games`, `scores`, `epoch`, `epoch_best`, `champion` и счётчики стагнации/рестартов.

## Вспомогательные функции (логика)
- `left_dir`, `right_dir`, `dir_after_action` — поворот/сохранение направления по действию 0/1/2.
- `state_key(game: &Game) -> u32` — компактный 20‑битный ключ состояния.
- `mutate_qagent(agent, rng, sigma)` — добавляет шум к значениям Q и затухает `epsilon`.
- `generate_population_colors(pop_size)`, `hsl_to_rgb`, `mutate_color` — палитра агентов.

## Визуализация (CPU)
- `Game::draw(frame)`, `draw_rect`, `draw_eyes` — отрисовка пиксельного буфера RGBA8.
- HUD и график в блоке `RedrawRequested` (внутри главного цикла). Кнопки: `draw_button` и хит‑тест `point_in_rect`.
- Функции низкого уровня: `clear_rgba`, `fill_cell_rgb`, `fill_rect_rgba`, `stroke_rect_rgba`, `blend_pixel`, `draw_text`, `draw_chart`, `draw_game_transparent` (все в `main.rs`).

## Вход/события
- Используется `WinitInputHelper`.
- Горячие клавиши: стрелки/WASD, P, R, E, H, +/-, G, U, B, J (DQN), K (NPU).
- Обработка мыши для нажатий на кнопки панели.

## GPU‑рендер (опционально, `gpu-render`)
- `gpu_render::GpuRenderer` — настройка `wgpu`, два пайплайна, инстансы клеток. Шейдеры:
  - `grid.wgsl` — фон‑«шахматка».
  - `instanced.wgsl` — инстансные квадраты (координаты gx/gy, цвет rgba).

## DQN (опционально, `dqn-gpu`)
- Модуль `dqn.rs`:
  - `DqnNet` — Embedding + 2xLinear + выход на 3 действия.
  - `DqnAgent` — сеть, оптимизатор AdamW, буфер `Replay`, параметры exploration.
  - Методы: `select_action`, `push_transition`, `train_step`, `save_safetensors`, `load_safetensors`.
  - `preferred_device()` — выбирает CUDA при наличии фичи `dqn-gpu-cuda`, иначе CPU.

## NPU/DirectML (опционально, `npu-directml`)
- `NpuPolicy` — загрузка ONNX‐модели, выбор действия через argmax над логитами.
- Вход — tensor int64 формы [1,1] с индексом состояния; выход — [1,3] логиты.

## Главная функция `main()`
- Создание окна, инициализация рендерера (pixels или GpuRenderer).
- Создание `Game` и `EvoTrainer` (популяция по умолчанию 24).
- Цикл событий: отрисовка, хэндлеры ввода, шаги эволюции батчами до `max_steps_per_tick`.
- Режимы GPU‑NN (выключено), DQN, NPU переключаются фичами и клавишами.

## Контракты/инварианты
- `Game::update` не выполняется, если `alive=false` или `paused=true`.
- В режиме без wrap попадание за границы — немедленная смерть (`DeathCause::Wall`).
- `EvoTrainer::reproduce` всегда восстанавливает размер популяции `pop_size` и вызывает `reset_epoch`.

## Возможные расширения
- Улучшить sampling в `Replay` (случайный батч, приоритетный replay).
- Перенести часть логики HUD/кнопок в отдельные модули.
- Добавить конфиг (toml) для гиперпараметров вместо констант.
