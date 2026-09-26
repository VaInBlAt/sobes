# Голосовой опрос (Flask)

Простое Flask-приложение: показывает по очереди вопросы из `test.csv`,
записывает ответ голосом прямо с телефона, распознаёт его через
`micro.py` (faster-whisper) и сравнивает с эталонным ответом с помощью
готовых методов `nli.py` и `paraphrase.py`.

## Структура

```
quizapp/
├── app.py              # Flask-сервер
├── micro.py            # транскрибация аудио (faster-whisper)
├── nli.py               # сравнение через NLI-модель
├── paraphrase.py        # сравнение через paraphrase-модель
├── test.csv             # question;correct_answer;answer1
├── requirements.txt
├── uploads_audio/       # временные аудиофайлы (удаляются после обработки)
└── templates/
    ├── index.html        # страница с вопросом и записью
    └── done.html          # экран после последнего вопроса
```

## Установка

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Если `nli.py` / `paraphrase.py` используют приватные модели на HuggingFace,
положите токен в `.env`:

```
HF_TOKEN=hf_xxx
```

## Запуск

```bash
python app.py
```

Сервер поднимется на `http://0.0.0.0:5000`.

## Доступ с телефона

Браузеры разрешают доступ к микрофону (`getUserMedia`) только по HTTPS
(или на `localhost`). Если открывать сайт с телефона просто по IP в
локальной сети (`http://192.168.x.x:5000`), микрофон работать не будет.

Варианты решения:
- прокинуть HTTPS-туннель, например `ngrok http 5000` или
  `cloudflared tunnel --url http://localhost:5000`, и открыть на
  телефоне выданный `https://...` адрес;
- либо развернуть приложение за настоящим HTTPS (nginx + сертификат).

## Как это работает

1. `GET /` — показывает текущий вопрос (индекс хранится в сессии).
2. Пользователь жмёт на кнопку 🎤, браузер записывает аудио через
   `MediaRecorder` (формат `webm/opus`).
3. По остановке записи файл отправляется на `POST /answer`.
4. Сервер сохраняет файл во временную папку, вызывает
   `micro.transcribe(...)`, получает текст, удаляет файл.
5. Распознанный текст сравнивается с `correct_answer` из `test.csv`:
   - `nli.compare_texts_nli(correct_answer, recognized_text)`
   - `paraphrase.compare_texts_paraphrase(correct_answer, recognized_text)`
6. Результат (тексты, оценки, бейджи ✓/✗ по порогу `0.8`) возвращается
   на страницу.
7. Кнопка «Следующий вопрос» дергает `POST /next`, сервер увеличивает
   индекс вопроса в сессии.
8. После последнего вопроса показывается экран `done.html` с кнопкой
   «Начать заново» (`POST /restart` обнуляет индекс).

## Настройка порогов

В `app.py`:

```python
NLI_THRESHOLD = 0.8
PARAPHRASE_THRESHOLD = 0.8
```

## Примечания

- `micro.transcribe` в `micro.py` по умолчанию распознаёт русский язык
  (`language="ru"`), это соответствует вопросам в `test.csv`.
- Модели (`faster-whisper`, NLI, paraphrase) загружаются один раз при
  старте сервера — первый запрос может занять время из-за инициализации.
- CSV читается с разделителем `;` (`pd.read_csv(CSV_PATH, sep=";")`),
  как и в исходном `main.py`.
