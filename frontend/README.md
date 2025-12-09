# Frontend - Система мониторинга вибрации подшипников

React приложение для визуализации данных вибрации подшипников электродвигателей.

## Установка зависимостей

```bash
npm install --legacy-peer-deps
```

или используйте скрипт:

```bash
npm run install-deps
```

## Запуск приложения

```bash
npm start
```

Приложение будет доступно по адресу: **http://localhost:3000**

**Важно:** Убедитесь, что вы открываете именно `http://localhost:3000`, а не другой адрес.

## Структура проекта

```
frontend/
├── public/
│   └── index.html          # HTML шаблон
├── src/
│   ├── components/
│   │   └── Dashboard.tsx   # Главный компонент Dashboard
│   ├── types/
│   │   └── vibration.ts    # TypeScript типы
│   ├── utils/
│   │   └── formatters.ts   # Утилиты форматирования
│   ├── App.tsx             # Главный компонент приложения
│   └── index.tsx           # Точка входа
├── package.json
└── tsconfig.json
```

## Возможные проблемы

### Страница не найдена

1. Убедитесь, что сервер разработки запущен (`npm start`)
2. Откройте браузер и перейдите на `http://localhost:3000`
3. Проверьте консоль браузера (F12) на наличие ошибок
4. Проверьте терминал, где запущен `npm start`, на наличие ошибок компиляции

### Ошибки компиляции

Если есть ошибки TypeScript или импортов:
1. Остановите сервер (Ctrl+C)
2. Удалите `node_modules` и `package-lock.json`
3. Выполните `npm install --legacy-peer-deps`
4. Запустите снова `npm start`

## Переменные окружения

Создайте файл `.env` в корне `frontend/`:

```
REACT_APP_API_URL=http://localhost:8000
```

## Технологии

- React 18
- TypeScript 4.9
- Material-UI (MUI) 5
- Chart.js для графиков
- Axios для HTTP запросов



