import numpy as np
from scipy.optimize import fmin_l_bfgs_b


def initial_trend(series, L):
    """
    Начальное значение для тренда.

    :param series: значения ряда
    :param L: длина сезона
    :return: начальное значение для тренда
    """
    sum = 0.0
    for i in range(L):
        sum += float(series[i + L] - series[i]) / L  # среднее средних трендов среди сезонов
    return sum / L


def initial_seasonal_components(series, L):
    """
    Начальные значения для seasonal components - Sx

    :param series: исходные значения ряда
    :param L: длина сезона
    :return: начальные значения для сезонов
    """
    seasonals = {}
    season_averages = []  # средние знячения в сезоне
    n = int(len(series) / L)  # количество сезонов
    for j in range(n):
        season_averages.append(sum(series[L * j:L * j + L]) / float(L))
    # каждое значение в сезоне делится на среднее по сезону,
    # затем полученный результат суммируется с соответствующим значением из следующего сезона,
    # полученная сумма делится на количество сезонов
    for i in range(L):
        sum_of_vals_over_avg = 0.0
        for j in range(n):
            sum_of_vals_over_avg += series[L * j + i] / season_averages[j]
        seasonals[i] = sum_of_vals_over_avg / n
    return seasonals


def holt_winters(series, L, alpha, beta, gamma, n_forecast):
    """
    Тройное экспоненциальное сглаживание.

    :param series: исходный ряд
    :param L: длина сезона
    :param alpha: параметр сглаживания для значения ряда
    :param beta: параметр сглаживания тренда
    :param gamma: сезонный параметр сглаживания
    :param n_forecast: количество точек, которые нужно предсказать
    :return: начальные значения + спрогнозированные
    """
    result = []
    for i in range(len(series) + n_forecast):
        if i == 0:  # начальные значения
            m = 1
            s = initial_seasonal_components(series, L)
            level = series[0]
            trend = initial_trend(series, L)
            result.append(series[0])
            continue
        if i >= len(series):  # прогноз
            m = i - len(series) + 1  # смещение в прогнозе
            # result.append(level + m * trend + s[i % L])
        else:  # HW на известных данных
            Yi = series[i]
            last_level = level  # baseline
            level = alpha * (Yi - s[i % L]) + (1 - alpha) * (level + trend)
            trend = beta * (level - last_level) + (1 - beta) * trend
            s[i % L] = gamma * (Yi - level) + (1 - gamma) * s[i % L]
            # result.append(Yi)
        result.append(level + m * trend + s[i % L])
    return result


# SSE - sum of squared residuals - сумма квадратов разностей
def SSE(coefficients, *arr):
    series = arr[0]
    season_length = arr[1]
    n_forecast = arr[2]
    alpha, beta, gamma = coefficients
    forecast = holt_winters(series, season_length,
                            alpha=alpha, beta=beta,
                            gamma=gamma, n_forecast=n_forecast)
    sse = 0
    for i in range(0, len(series)):
        sse += (forecast[i] - series[i]) ** 2
    return sse


def forecast(series, season_length, n_forecast):
    Y = series[:]
    initial_values = np.array([0.3, 0.1, 0.1])
    boundaries = [(0, 1), (0, 1), (0, 1)]
    # минимизируем ошибку с помощью алгоритма L-BFGS-B
    parameters = fmin_l_bfgs_b(SSE, x0=initial_values,
                               args=(Y, season_length, n_forecast),
                               bounds=boundaries,
                               approx_grad=True)
    alpha, beta, gamma = parameters[0]
    print(parameters[0])
    return holt_winters(series, season_length, alpha, beta, gamma, n_forecast)
