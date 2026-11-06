#!/bin/bash

# Переходим в директорию, где лежит этот скрипт
cd "$(dirname "$0")"

# Автоматически активируем наше чистое виртуальное окружение
source venv/bin/activate

# Запускаем наше алхимическое ядро Avery Edition
echo "[Avery Core] Запускаю Foooocus-Lite v0.03..."
python3 -m uvicorn main:app --reload --host 127.0.0.1 --port 8000


