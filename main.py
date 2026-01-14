"""
Главный файл для запуска Telegram-бота
"""

import logging
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    filters,
)

from bot.botstocks import Botstocks, TICKER, AMOUNT
from utils.logger import setup_logging
from config import config

# ===== Настройка логирования =====
setup_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    """Точка входа в приложение"""

    logger.info("=" * 60)
    logger.info("🚀 Запуск Telegram-бота для прогнозирования акций")
    logger.info("=" * 60)

    # ===== Проверка токена =====
    if not config.BOT_TOKEN or config.BOT_TOKEN == "YOUR_BOT_TOKEN":
        logger.critical("BOT_TOKEN не найден!")
        logger.critical(
            "Создайте файл .env и добавьте строку:\n"
            "BOT_TOKEN=ваш_токен_бота"
        )
        return

    # ===== Создание приложения =====
    application = Application.builder().token(config.BOT_TOKEN).build()

    # ===== Обработчик диалога =====
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", Botstocks.start)],
        states={
            TICKER: [
                MessageHandler(
                    filters.TEXT & ~filters.COMMAND,
                    Botstocks.ticker_received
                )
            ],
            AMOUNT: [
                MessageHandler(
                    filters.TEXT & ~filters.COMMAND,
                    Botstocks.amount_received
                )
            ],
        },
        fallbacks=[CommandHandler("cancel", Botstocks.cancel)],
        name="stock_forecast_conversation",
        persistent=False,
    )

    # ===== Регистрация обработчиков =====
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler("help", Botstocks.help_command))

    # ===== Информация о запуске =====
    logger.info("✅ Бот успешно запущен")
    logger.info("📈 Горизонт прогноза: %d дней", config.FORECAST_DAYS)
    logger.info("📊 Исторические данные: %d дней", config.HISTORY_DAYS)
    logger.info("-" * 60)

    # ===== Запуск polling =====
    application.run_polling(
        allowed_updates=Update.ALL_TYPES
    )


if __name__ == "__main__":
    main()
