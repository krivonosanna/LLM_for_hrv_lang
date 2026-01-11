# 🇭🇷 HR-LLM: Generative LLM for Croatian Language

> Large Language Model, специализирующаяся на **генерации текста на хорватском языке**.

## 🎯 Бизнес-цель

Создать надёжный и качественный генеративный ИИ для хорватского языка, который можно внедрить в:
- редакторы контента (автодополнение, черновики статей),
- маркетинговые инструменты (генерация описаний товаров, email-рассылок),
- образовательные платформы (помощь в написании эссе),
- корпоративные чат-боты (естественные ответы на запросы).


## 🗃️ Набор данных

Для обучения и валидации используется [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) для хорватского языка. 

Данные используются для:
- Предобучения (pre-training)
- Оценки качества (held-out test set)

## 📈 Целевые метрики

### Производительность:
- Время отклика (≤150 токенов): **≤300 мс** (среднее), **≤600 мс** (P95)
- Доля неуспешных запросов: **≤1%**
- Использование GPU/CPU и памяти укладывается в типичные лимиты бесплатного тарифа Google Colab (до 13 ГБ RAM, 12–80 ГБ VRAM в зависимости от выделенного GPU).

### Качество генерации:
- Perplexity на тестовом хорватском корпусе: **≤15**
- Loss на тестовом хорватском корпусе: **≤5**


## 🧪 План экспериментов

| Этап | Описание | Инструменты |
|------|-----------|-------------|
| **1. Исследование данных** | Анализ структуры, длины и частотности текста | `transformers`, `datasets`|
| **2. Предобработка** | Очистка, токенизация, нормализация | `BPE`, `clean-text` |
| **3. Обучение модели** | Выбор архетиктуры и обучение модели | `transformers`, `PyTorch` |
| **4. Оценка качества** | Расчёт Perplexity и Loss на валидации | `transformers`, `PyTorch` |
| **6. Развёртывание** | Логирование результатов и тестирование| `logging`, `Hugging Face`, `GitHub Actions` |

<!-- ; ## 🛠️ Технологии
; - Hugging Face Transformers, PyTorch
; - Weights & Biases (трекинг экспериментов)
; - Docker, Prometheus, Grafana
; - Python, FastAPI (для API) -->

## ⚙️ Технологический стек

- **Язык:** Python 3.11  
- **Фреймворки:** PyTorch, Hugging Face Transformers  
<!-- ; - **API:** FastAPI  
; - **Мониторинг:** Prometheus + Grafana  
; - **Инфраструктура:** Docker, Kubernetes   -->
- **CI/CD:** GitHub Actions 

<!-- ## Вызов

Для запуска обучения модели используйте следующую команду:

```bash
pip install -r requirements.txt
python train.py --config config.yaml
```

Для быстрого запуска (только проверить работоспособность)

```bash
pip install -r requirements.txt
python train.py --config config.yaml data.num_samples=1000 tokenizer.num_samples_for_tokenizer=1000 trainer.n_steps=5 trainer.val_every_n_steps=3 trainer.plot_every_n_steps=1
``` -->


## Вызов
Данные и модель версионируются с помощью DVC (Data Version Control) и физически находятся в удалённом хранилище на DAGsHub. 

Для клонировния:

```bash
git clone https://github.com/krivonosanna/LLM_for_hrv_lang.git
```

Для загрузки данных и модели (для подключения нужен токен из .dvc - добавлен для возможности проверки):

```bash
pip install -r requirements.txt
export DAGSHUB_USERNAME=USERNAME
export DAGSHUB_TOKEN=ваш_токен_из_dagshub
dvc remote add origin https://dagshub.com/${DAGSHUB_USERNAME}/LLM_for_hrv_lang.dvc
dvc remote modify origin auth basic
dvc remote modify origin user "${DAGSHUB_USERNAME}"
dvc remote modify origin password "${DAGSHUB_TOKEN}"
dvc pull 
```

Если нужно переобучить модель:

```bash
pip install -r requirements.txt 
dvc repro 
```

## 📈 Трекинг экспериментов с MLflow

Все запуски обучения автоматически логируются в MLflow — систему для управления жизненным циклом машинного обучения.

По умолчанию все данные (параметры, метрики, артефакты) сохраняются локально в папку:

```
./mlruns/
```

Чтобы открыть веб-интерфейс:

```bash
mlflow ui
```

## 🐳 Docker-образ

Сборка 

```bash
docker build -t ml-app:v1 .    
```

Запуск

```bash
docker run --rm -v $(pwd):/data ml-app:v1 --input_path /data/input_model.csv --output_path /data/outputs.csv
```

Формат входа - CSV с колонкой input_model:

```csv
input_model
"Zagreb je glavni grad Hrvatske."
"Kava je popularna u Hrvatskoj."
```

Формат выхода - CSV с колонками input (исходный текст) и prediction (сгенерированный моделью текст на хорватском языке):

```csv
input,prediction
"Zagreb je glavni grad Hrvatske.","Zagreb je glavni grad Hrvatske i najveći grad u zemlji po broju stanovnika."
```

📦 Что делает скрипт src/predict.py?

- Загружает предобученную causal language model и токенайзер из локальных папок (model/, tokenizer/)
- Читает входной CSV-файл по пути --input_path
- Для каждой строки в колонке input_model генерирует продолжение
- Сохраняет пары (исходный текст, предсказание) в CSV по пути --output_path

## Развёртывание модели как онлайн-сервиса с TorchServe

Сохраните модель:

```bash
python export_model.py
```

Переместите tokenizer в отдельную папку - extra-files;

Подготовка архива модели:

```bash
mkdir -p serve/model-store

torch-model-archiver \                                  
  --model-name mymodel \
  -v 1.0 \
  --serialized-file model.pt \
  --handler src/handler.py \
  --extra-files "extra-files/" \
  -r requirements_serve.txt \
  --export-path serve/model-store \
  -f
  ```

Сборка:

```bash
docker build -f Dockerfile.serve -t mymodel-serve:v1 .  
```

Запуск:

```bash
docker run -d \
  --name hrv-llm-service \
  -p 8080:8080 \
  -p 8081:8081 \
  -p 8082:8082 \
  mymodel-serve:v1
```

Отправьте запрос:

```bash
curl -X POST http://localhost:8080/predictions/mymodel \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text"}'
```


## ✅ Контакты

**Автор:** *[Кривонос Анна]* 
**Email:** [annay3294@gmail.com]  

<!-- ; ## 📜 Лицензия
; Проект использует только открытые модели и данные с совместимыми лицензиями (например, Llama-3 с Meta License, данные — CC-BY/CC0). -->