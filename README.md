# CSMP — Compressive Sensing Matching Pursuit

Библиотека **CSMP** предоставляет инструменты для сжатия и восстановления сигналов с использованием методов разреженного представления (Compressive Sensing).

---

## 🚀 Установка

```bash
git clone https://github.com/xephosbot/CompressiveSensing.git
cd csmp
pip install .
```

---

## 🧠 Поддерживаемые алгоритмы

- `Matching Pursuit (MP)`
- `Orthogonal Matching Pursuit (OMP)`

---

## 📦 Пример использования

```python
import csmp

# Генерация сигнала
_, x = csmp.generate_signal(4096)

# Сжатие сигнала
y, Theta, _ = csmp.compress_signal(x, 500)

# Восстановление с помощью OMP
s_hat = csmp.orthogonal_matching_pursuit(Theta, y, K=10)

# Реконструкция сигнала
x_hat = csmp.reconstruct_signal(s_hat)

# Оценка качества
print("SNR:", csmp.calculate_snr(x, x_hat))
```

---

## 📊 Метрики

- Signal-to-Noise Ratio (SNR)
- Mean Square Error (MSE)
- Mean Absolute Error (MAE)

---
