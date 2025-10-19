# Экспорт весов DQN (safetensors → ONNX)

Этот скрипт конвертирует веса DQN из формата `.safetensors` (Candle) в ONNX, пригодный для инференса через ONNX Runtime / DirectML (NPU).

## Структура сети
- Embedding(vocab → hidden)
- Linear(hidden → hidden) + ReLU
- Linear(hidden → hidden) + ReLU
- Linear(hidden → actions)

Вход: индекс состояния `state_idx` формы `[B,1]` (int64)
Выход: `logits` формы `[B, actions]`

## Установка зависимостей
```powershell
pip install -r src/export/requirements.txt
```

## Экспорт
```powershell
python src/export/export_to_onnx.py --safetensors dqn_agent.safetensors --vocab 1024 --hidden 256 --actions 3 --out snake_dqn.onnx
```

Флажи:
- `--transpose-linear` — если формы весов линейных слоёв не совпадают, скрипт попробует транспонировать матрицы.
- `--opset` — версия opset ONNX (по умолчанию 17).
- `--no-verify` — пропустить проверку onnxruntime после экспорта.

## Где использовать модель
- Приложение ищет путь в `SNAKE_NPU_MODEL` (или совместимый `SNAKE_NPU_ONNX`) и поддерживает `.onnx` и `.ort`.
- Можно положить файл в одну из стандартных папок: `./`, `models/`, `assets/`, `target/release/`, `target/debug/`.

## Проверка onnxruntime
Скрипт по умолчанию пробует выполнить один прогон на CPU (ORT), чтобы проверить форму и корректность вывода.

## Советы
- `vocab`, `hidden`, `actions` должны совпадать с параметрами модели, использованными при обучении.
- Если имена ключей в safetensors отличаются от ожидаемых (emb/mlp1/mlp2/out), скрипт выведет список доступных ключей и попытается найти подходящий. При необходимости поправьте поиск в скрипте.
