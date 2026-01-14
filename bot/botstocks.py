import os
import logging
from telegram import Update
from telegram.ext import ContextTypes, ConversationHandler

from services.data_service import DataService
from services.prediction_service import PredictionService
from services.visualization_service import VisualizationService
from utils.trading_signals import TradingSignals
from utils.logger import log_user_request
from config import config

logger = logging.getLogger(__name__)

# ===== Константы =====
TICKER, AMOUNT = range(2)
MAX_INVESTMENT = 1_000_000_000
PARSE_MODE = 'HTML'


class Botstocks:
    """Класс с обработчиками Telegram-бота"""

    @staticmethod
    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Начало диалога"""
        await update.message.reply_text(
            "👋 <b>Добро пожаловать!</b>\n\n"
            "Я бот для прогнозирования цен акций  "
           
            "⚠️ <b>Важно:</b> Я создан в учебных целях и не является "
            "финансовой рекомендацией.\n\n"
            "Я использую исторические данные и построю прогноз на "
            f"{config.FORECAST_DAYS} дней с помощью трех разных моделей:\n"
            "• Random Forest \n"
            "• ARIMA \n"
            "• LSTM (нейросеть) \n\n"
            "Введите тикер компании (например, AAPL, MSFT, TSLA):",
            parse_mode=PARSE_MODE
        )
        return TICKER

    @staticmethod
    async def ticker_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Обработка тикера"""
        ticker = update.message.text.strip().upper()

        if not DataService.validate_ticker(ticker):
            await update.message.reply_text(
                "❌ Некорректный тикер.\n"
                "Введите валидный тикер (например, AAPL, MSFT, GOOGL):"
            )
            return TICKER

        context.user_data['ticker'] = ticker

        await update.message.reply_text(
            f"✅ Тикер: <b>{ticker}</b>\n\n"
            "Введите сумму для инвестиции в долларах "
            "(например, 777):",
            parse_mode=PARSE_MODE
        )
        return AMOUNT

    @staticmethod
    async def amount_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Обработка суммы и запуск анализа"""
        try:
            amount = Botstocks._parse_amount(update.message.text.strip())
            
            if not Botstocks._validate_amount(amount):
                await update.message.reply_text(
                    Botstocks._get_amount_error_message(amount)
                )
                return AMOUNT

            context.user_data['amount'] = amount
            ticker = context.user_data['ticker']

            Botstocks._log_analysis_start(update, ticker, amount)
            
            await update.message.reply_text(
                f" <b>Начинаю анализ акций {ticker}</b>\n\n"
                " Загружаю данные за последние 2 года...\n"
                " Обучаю модели машинного обучения...\n"
                " Строю прогноз...\n\n"
                " Это займет 60 минут, пожалуйста, подождите...",
                parse_mode=PARSE_MODE
            )

            # ===== Загрузка данных =====
            data_service = DataService()
            data = data_service.load_stock_data(ticker)

            if data is None:
                await update.message.reply_text(
                    f"❌ <b>Ошибка загрузки данных</b>\n\n"
                    f"Не удалось загрузить данные для тикера <b>{ticker}</b>.\n"
                    "Возможные причины:\n"
                    "• Неверный тикер\n"
                    "• Проблемы с Yahoo Finance\n"
                    "• Тикер не торгуется\n\n"
                    "Используйте /start для новой попытки.",
                    parse_mode=PARSE_MODE
                )
                return ConversationHandler.END

            # ===== Обучение и прогноз =====
            prediction_service = PredictionService()
            prediction_service.train_all_models(data)
            predictions = prediction_service.predict(steps=config.FORECAST_DAYS)

            # ===== Торговые сигналы =====
            trading_signals = TradingSignals()
            buy_days, sell_days = trading_signals.find_extrema(predictions)
            profit, strategy = trading_signals.calculate_profit(
                predictions, amount, buy_days, sell_days
            )

            # ===== Визуализация =====
            viz_service = VisualizationService()
            chart_file = viz_service.plot_prediction(
                ticker, data, predictions, buy_days, sell_days
            )

            await Botstocks._send_chart(update, chart_file)

            # ===== Формирование отчета =====
            results = prediction_service.get_results_summary()
            current_price = data['price'].iloc[-1]
            predicted_price = predictions[-1]
            price_change = ((predicted_price - current_price) / current_price) * 100

            trend_emoji = "📈" if price_change > 0 else "📉"
            trend_text = "вырастет" if price_change > 0 else "упадет"

            report = Botstocks._build_report(
                ticker, results, current_price, predicted_price,
                price_change, trend_emoji, trend_text, 
                amount, profit, strategy
            )

            await update.message.reply_text(report, parse_mode=PARSE_MODE)

            # ===== Логирование =====
            Botstocks._log_analysis_complete(
                update, ticker, amount, 
                results['best_model'], results['best_rmse'], profit
            )

        except ValueError:
            await update.message.reply_text(
                "❌ Некорректное число. Введите сумму (например, 10000):"
            )
            return AMOUNT

        except Exception as e:
            logger.error("Ошибка при обработке запроса: %s", str(e), exc_info=True)
            await Botstocks._handle_error(update)

        return ConversationHandler.END

    @staticmethod
    def _parse_amount(text: str) -> float:
        """Парсит сумму из текста"""
        return float(text.strip().replace(',', ''))

    @staticmethod
    def _validate_amount(amount: float) -> bool:
        """Проверяет корректность суммы"""
        return 0 < amount <= MAX_INVESTMENT

    @staticmethod
    def _get_amount_error_message(amount: float) -> str:
        """Возвращает сообщение об ошибке суммы"""
        if amount <= 0:
            return "❌ Сумма должна быть положительной. Попробуйте еще раз:"
        return "❌ Сумма слишком большая. Введите реальную сумму:"

    @staticmethod
    async def _send_chart(update: Update, chart_file: str):
        """Отправляет график и удаляет временный файл"""
        try:
            with open(chart_file, 'rb') as photo:
                await update.message.reply_photo(photo=photo)
        finally:
            try:
                os.remove(chart_file)
            except OSError as e:
                logger.warning("Не удалось удалить временный файл %s: %s", chart_file, e)

    @staticmethod
    def _log_analysis_start(update: Update, ticker: str, amount: float):
        """Логирует начало анализа"""
        logger.info(
            "Начат анализ | user_id=%s | ticker=%s | amount=%.2f",
            update.effective_user.id,
            ticker,
            amount
        )

    @staticmethod
    def _build_report(ticker: str, results: dict, current_price: float, 
                     predicted_price: float, price_change: float, 
                     trend_emoji: str, trend_text: str, amount: float, 
                     profit: float, strategy: str) -> str:
        """Строит финальный отчет"""
        roi = (profit / amount) * 100 if amount else 0
        
        report_parts = [
            f" <b>ОТЧЕТ ПО АКЦИЯМ {ticker}</b>",
            "=" * 40,
            "",
            " <b>Модели машинного обучения:</b>",
        ]
        
        for model_name, rmse in results['all_results'].items():
            if rmse == float('inf'):
                report_parts.append(f"• {model_name}: ❌ Ошибка обучения")
            else:
                star = " ⭐" if model_name == results['best_model'] else ""
                report_parts.append(f"• {model_name}: RMSE = {rmse:.2f}{star}")
        
        report_parts.extend([
            "",
            f" <b>Лучшая модель:</b> {results['best_model']}",
            f" <b>RMSE:</b> {results['best_rmse']:.2f}",
            "",
            "=" * 40,
            " <b>АНАЛИЗ ЦЕН:</b>",
            f"• Текущая цена: <b>${current_price:.2f}</b>",
            f"• Прогноз: <b>${predicted_price:.2f}</b>",
            f"• Изменение: {trend_emoji} <b>{abs(price_change):.2f}%</b> ({trend_text})",
            "",
            "=" * 40,
            " <b>ИНВЕСТИЦИОННАЯ СТРАТЕГИЯ:</b>",
            f"• Инвестиция: <b>${amount:,.2f}</b>",
            f"• Прибыль: <b>${profit:,.2f}</b>",
            f"• ROI: <b>{roi:.2f}%</b>",
            "",
            "=" * 40,
           
            " <b>Предупреждение:</b>",
            "Прогноз носит исключительно учебный характер.",
            "",
            "Используйте /start для нового анализа."
        ])
        
        return "\n".join(report_parts)

    @staticmethod
    def _log_analysis_complete(update: Update, ticker: str, amount: float,
                              best_model: str, best_rmse: float, profit: float):
        """Логирует завершение анализа"""
        logger.info(
            "Анализ завершён | user_id=%s | ticker=%s | profit=%.2f",
            update.effective_user.id,
            ticker,
            profit
        )
        
        log_user_request(
            user_id=update.effective_user.id,
            ticker=ticker,
            amount=amount,
            model=best_model,
            metric=best_rmse,
            profit=profit
        )

    @staticmethod
    async def _handle_error(update: Update):
        """Обрабатывает общую ошибку"""
        await update.message.reply_text(
            "❌ <b>Произошла ошибка</b>\n\n"
            "Попробуйте:\n"
            "• Проверить тикер\n"
            "• Выбрать другую компанию\n"
            "• Повторить позже\n\n"
            "Используйте /start для новой попытки.",
            parse_mode=PARSE_MODE
        )

    @staticmethod
    async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Отмена диалога"""
        await update.message.reply_text(
            "❌ Операция отменена.\n\nИспользуйте /start для нового анализа."
        )
        return ConversationHandler.END

    @staticmethod
    async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда помощи"""
        await update.message.reply_text(
            "📖 <b>СПРАВКА</b>\n\n"
            "/start — начать анализ\n"
            "/cancel — отменить операцию\n\n"
            "Введите тикер и сумму, чтобы получить прогноз и рекомендации.",
            parse_mode=PARSE_MODE
        )


# ===== Экспорт для обратной совместимости =====
__all__ = ['Botstocks', 'TICKER', 'AMOUNT']