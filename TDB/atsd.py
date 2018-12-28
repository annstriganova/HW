from atsd_client import connect
from atsd_client.models import *
from atsd_client.services import SeriesService
import matplotlib.pyplot as plt

from TDB.hw import forecast

# В файле connection.properties нужно указать хост, имя и пароль от ATSD
connection = connect();
# Указываем интересующие нас метрику и сущность, для которых будет строиться прогноз
sf = SeriesFilter(metric="direct.queries")
ef = EntityFilter(entity="duckduckgo")
# Выбираем начальную дату для запроса
start_date = "2018-11-03T02:59:00Z"
# Инициализируем временной фильтр, конечная дата = начальная дата + 3 месяца
df = DateFilter(interval={"count": 3, "unit": "MONTH"}, start_date=start_date)
# Формируем series-запрос
query_data = SeriesQuery(series_filter=sf, entity_filter=ef, date_filter=df)
svc = SeriesService(connection)
# Загружаем указанный ряд
series, = svc.query(query_data)
"""
Визуализируем исторические и спрогнозированные данные
"""
print(series)
start = len(series.values())
end = start + 25
plt.subplot(111)
plt.plot(series.values(), label="History", marker=".")
plt.plot(range(start, end), forecast(series.values(), 7, 25)[-25:], label="Forecast", marker=".")
plt.legend(bbox_to_anchor=(1, 0.14), loc=1, borderaxespad=0.1)
plt.grid(True)
plt.show()
