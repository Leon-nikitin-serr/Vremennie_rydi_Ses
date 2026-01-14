Телеграм-бот для анализа и прогнозирования акций на основе временных рядов

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Telegram](https://img.shields.io/badge/Telegram-Bot-blue)](https://core.telegram.org/bots)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


 Возможности
 Несколько ML-моделей: Три различных алгоритма прогнозирования

 Интерактивная визуализация: Профессиональные графики с сигналами покупки/продажи

 Торговые рекомендации: Практические инвестиционные идеи

 Анализ в реальном времени: Интеграция с Yahoo Finance

Прогнозирование временных рядов: Прогноз цен на 30 дней вперед

 Используемые модели машинного обучения
Модель	Тип	Описание
Random Forest	Ансамблевое обучение	С лаговыми признаками и скользящими средними
ARIMA	Статистическая	Классический анализ временных рядов
LSTM	Глубокое обучение	Нейронная сеть для прогнозирования последовательностей

<pre>
TGbot/
├── bot/
│   └── botstocks.py             # Обработчики Telegram-бота
├── models/                      # Модели машинного обучения
│   ├── base_model.py           # Базовый класс моделей
│   ├── random_forest.py        # Random Forest модель
│   ├── arima_model.py          # ARIMA модель
│   └── lstm_model.py           # LSTM нейронная сеть
├── services/                    # Сервисы приложения
│   ├── data_service.py         # Сервис работы с данными
│   ├── prediction_service.py   # Сервис прогнозирования
│   └── visualization_service.py # Сервис визуализации
├── utils/                       # Вспомогательные утилиты
│   ├── logger.py               # Настройка логирования
│   └── trading_signals.py      # Определение торговых сигналов
├── config.py                    # Конфигурация приложения
├── main.py                      # Главный файл запуска
├── requirements.txt             # Зависимости Python
├── README.md                    # Документация
└── .gitignore                   # Исключаемые файлы Git

Логи (автогенерация):
├── logs.txt                    # Детальные логи приложения
└── bot.log                     # Логи пользовательских запросов
</pre>

