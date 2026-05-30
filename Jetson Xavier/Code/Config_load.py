
import os
import sys 
import json
import re

# Загрузка конфига с поддержкой комментариев (JSONC)

def load_config(path="config.jsonc"):
    if not os.path.exists(path):
        print(f"Файл конфигурации '{path}' не найден.")
        sys.exit(1)
    file = open(path, 'r', encoding='utf-8')
    content = file.read()
    file.close()
    # Удаляем однострочные комментарии вида // ...
    content = re.sub(r'//.*', '', content)
    return json.loads(content)