# Инструкции для Claude Code

## Работа с Python-зависимостями

**ВАЖНО**: Всегда используй виртуальное окружение (venv) для установки зависимостей и запуска Python-скриптов.

### Правила работы с venv:

1. **Перед установкой зависимостей** - убедись, что venv активирован
2. **Перед запуском скриптов** - всегда используй Python из venv
3. **НИКОГДА** не устанавливай пакеты в системный Python

### Команды:

```bash
# Создание venv (если еще не создан)
python3 -m venv venv

# Активация venv
source venv/bin/activate  # для macOS/Linux
# или
venv\Scripts\activate     # для Windows

# Установка зависимостей
pip install -r requirements.txt

# Запуск скриптов
python script.py
```

### Альтернативный подход (без активации):

```bash
# Установка зависимостей через venv
venv/bin/pip install package_name

# Запуск скриптов через venv
venv/bin/python script.py
```
