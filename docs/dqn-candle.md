# DQN на Candle — руководство

Модуль `src/dqn.rs` активируется фичей `dqn-gpu`. Он реализует Deep Q‑Network для выбора действий в среде змейки.

## Архитектура сети
- Embedding: `state_vocab -> hidden`
- MLP: `hidden -> hidden -> hidden`
- Выход: `hidden -> 3` (3 действия)
- Активации ReLU.

## Параметры агента
- `input_vocab` — размер дискретного пространства состояний (пример: 1024). Состояния мэпятся как `state % input_vocab`.
- `epsilon`, `min_epsilon`, `decay` — ε‑жадная политика.
- `gamma` — коэффициент дисконтирования.
- Буфер повторов `Replay` вместимостью 20k.

## Обучение
- Потеря: MSE между `Q(s,a)` и `r + γ max_a' Q(ns,a')` c маской окончаний эпизодов.
- Батч-сэмплинг пока простой (первые `B`), можно улучшить до случайного.
- Оптимизатор: `AdamW` (`candle_nn`).

## Устройства
- По умолчанию CPU. При фиче `dqn-gpu-cuda` и доступном устройстве — CUDA.

## Работа с весами
- Сохранение: `.save_safetensors(path)`
- Загрузка: `.load_safetensors(path)`
- Файл по умолчанию: `dqn_agent.safetensors` в корне проекта.

## Запуск

- CPU:
  ```powershell
  cargo run --features dqn-gpu
  ```
- CUDA:
  ```powershell
  cargo run --features "dqn-gpu dqn-gpu-cuda"
  ```

В приложении нажмите J для включения/выключения DQN. Эволюция (E) должна быть включена, чтобы DQN участвовал в шагах. В DQN‑режиме мир автоматически переключается на «твёрдые» стены.

## Настройка загрузки на GPU (CUDA)

По умолчанию сеть и батч умеренных размеров. Чтобы увеличить загрузку GPU/VRAM, можно управлять параметрами через переменные окружения:

- `SNAKE_DQN_VOCAB` — размер словаря состояний (default: 1024)
- `SNAKE_DQN_HIDDEN` — размер скрытого слоя (default: 256)
- `SNAKE_DQN_BATCH` — размер батча тренировки (default: 256)

Пример (PowerShell):

```powershell
$env:SNAKE_DQN_VOCAB=4096; $env:SNAKE_DQN_HIDDEN=512; $env:SNAKE_DQN_BATCH=1024; cargo run --features "dqn-gpu dqn-gpu-cuda"
```

Подсказки:
- Увеличение `HIDDEN` и `BATCH` обычно сильнее нагружает VRAM и тензорные ядра.
- Слишком большие значения могут замедлить UI; при необходимости включайте Ultra (U) или показывайте только лучшего агента (B).
