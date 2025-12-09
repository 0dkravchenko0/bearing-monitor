# Решение проблем

## Проблема: "Страница не найдена"

### Шаг 1: Проверьте URL
Убедитесь, что вы открываете правильный адрес:
- ✅ **Правильно:** `http://localhost:3000`
- ❌ **Неправильно:** `http://localhost:3000/some-path` или другой адрес

### Шаг 2: Проверьте, что сервер запущен
В терминале должно быть сообщение:
```
Compiled successfully!

You can now view bearing-monitor-frontend in the browser.

  Local:            http://localhost:3000
```

### Шаг 3: Проверьте консоль браузера
1. Откройте DevTools (F12)
2. Перейдите на вкладку "Console"
3. Проверьте наличие ошибок (красные сообщения)

### Шаг 4: Проверьте терминал с npm start
Ищите ошибки компиляции, например:
- `Failed to compile`
- `Module not found`
- `Type errors`

### Шаг 5: Очистка и переустановка
Если есть ошибки компиляции:

```bash
# Остановите сервер (Ctrl+C)

# Удалите node_modules и package-lock.json
Remove-Item -Recurse -Force node_modules
Remove-Item package-lock.json

# Переустановите зависимости
npm install --legacy-peer-deps

# Запустите снова
npm start
```

### Шаг 6: Проверьте порт
Если порт 3000 занят, React автоматически предложит использовать другой порт (например, 3001).
Следуйте инструкциям в терминале.

## Частые ошибки

### Ошибка: "Cannot find module"
**Решение:** Удалите `node_modules` и `package-lock.json`, затем выполните `npm install --legacy-peer-deps`

### Ошибка: "Type errors"
**Решение:** Проверьте версию TypeScript - должна быть 4.9.5 в `package.json`

### Ошибка: "Chart.js not found"
**Решение:** Убедитесь, что установлены все зависимости:
```bash
npm install --legacy-peer-deps
```

### Браузер показывает пустую страницу
1. Проверьте консоль браузера (F12)
2. Проверьте вкладку "Network" - все ли файлы загрузились
3. Попробуйте очистить кэш браузера (Ctrl+Shift+Delete)



