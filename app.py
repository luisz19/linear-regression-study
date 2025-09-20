x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

def mean(values):
    return sum(values) / len(values)

def variance(values):
    m = mean(values)
    return sum((x - m) ** 2 for x in values)

def covariance(x, y):
    mx, my = mean(x), mean(y)
    return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))

def predict(x):
    return b0 + b1 * x

b1 = covariance(x, y) / variance(x)

b0 = mean(y) - b1 * mean(x)

y_pred = [predict(xi) for xi in x]

mse = sum((yi - ypi) ** 2 for yi, ypi in zip(y, y_pred)) / len(y)
r2 = 1 - sum((yi - ypi) ** 2 for yi, ypi in zip(y, y_pred)) / sum((yi - mean(y)) ** 2 for yi in y)

for xi, yi in zip(x, y):
   print(f"horas estudadas: {xi}, nota obtida: {yi}")

print(f"Média: xm = {mean(x)}, ym = {mean(y)}")
print(f"Variância: Var(x) = {variance(x)}")
print(f"Covariância: Cov(x, y) = {covariance(x, y)}")
print(f"Coeficiente de inclinação (b1): {b1}")
print(f"Coeficiente intercepto (b0): {b0}")
print(f"Valores preditos: {y_pred}")
print(f"Erro Quadrático Médio (MSE): {mse}")
print(f"Coeficiente de Determinação (R²): {r2}")
