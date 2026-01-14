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

TGbot_github/
├── bot/
│   └── botstocks.py
├── models/
│   ├── base_model.py
│   ├── random_forest.py
│   ├── arima_model.py
│   └── lstm_model.py
├── services/
│   ├── data_service.py
│   ├── prediction_service.py
│   └── visualization_service.py
├── utils/
│   ├── logger.py
│   └── trading_signals.py
├── config.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore

Логи:
├── logs.txt                    # Логи приложения
└── bot.log                     # Логи пользовательских запросов
