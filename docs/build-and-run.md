# Сборка и запуск (Windows)

Эта памятка фокусируется на Windows PowerShell и поддерживаемых фичах.

## Предварительные требования
- Rust (stable): https://rustup.rs
- Графические драйверы (для `wgpu`/DirectX).
- (Опционально) CUDA Toolkit + драйвер NVIDIA для `dqn-gpu-cuda`.
- (Опционально) ONNX Runtime DirectML — скачивание бинарей управляется crate `ort` при сборке.

## Базовый запуск

```powershell
cargo run
```

Оптимизированный:

```powershell
cargo run --release
```

## Полезные фичи

- CPU‑рендер (по умолчанию `pixels`) — ничего включать не нужно.
- GPU‑рендер через `wgpu`:
  ```powershell
  cargo run --features gpu-render
  ```
- DQN (Candle) на CPU:
  ```powershell
  cargo run --features dqn-gpu
  ```
- DQN с CUDA:
  ```powershell
  cargo run --features "dqn-gpu dqn-gpu-cuda"
  ```
- NPU (DirectML, ONNX):
  ```powershell
  $env:SNAKE_NPU_ONNX = "C:\\path\\to\\snake_dqn.onnx"; cargo run --features npu-directml
  ```

## Частые ошибки и решения
- «No GPU adapter» при `gpu-render` — используйте CPU‑рендер (без фичи), проверьте драйверы.
- DQN CUDA: «Device init failed» — проверьте корректность установки CUDA и совместимость версий с `candle-core`.
- NPU: «ONNX model not found» — задайте `SNAKE_NPU_ONNX` или поместите `snake_dqn.onnx` рядом с бинарником (см. README).
- Низкий FPS при больших `steps/frame` — включите режим Ultra (U) или `BEST` (B), чтобы снизить нагрузку на отрисовку.
