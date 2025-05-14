# CSMP Framework - Документация

## Введение

CSMP Framework - это библиотека для работы с методом Compressive Sensing (CS) для сжатия и восстановления 1D сигналов. Название CSMP расшифровывается как **C**ompressive **S**ensing with **M**atching **P**ursuit.

Метод Compressive Sensing позволяет восстанавливать сигнал из меньшего числа отсчётов по сравнению с классическими методами дискретизации. Это возможно благодаря использованию разреженного представления сигнала в определенном базисе и специальных алгоритмов восстановления, таких как Matching Pursuit (MP), Orthogonal Matching Pursuit (OMP).

## Основные компоненты

### Базисы (Basis)

В библиотеке реализованы базисы для разреженного представления сигналов:

- `DCTBasis` - дискретное косинусное преобразование (ДКП)
- `DFTBasis` - дискретное преобразование Фурье (ДПФ)

Каждый базис обеспечивает методы прямого и обратного преобразования сигнала, а также построение матрицы базиса для сигнала заданной длины.

### Матрица выбора отсчетов (SamplingMatrix)

Класс `SamplingMatrix` предоставляет статические методы для создания различных типов матриц выбора отсчетов:

- `random_rows` - выбор случайных строк (отсчетов) сигнала
- `gaussian` - матрица с элементами из нормального распределения
- `bernoulli` - матрица с элементами из распределения Бернулли

### Алгоритмы восстановления (ReconstructionAlgorithm)

Реализованы следующие алгоритмы восстановления:

- `MP` (Matching Pursuit) - итеративный алгоритм, который находит разреженное представление сигнала
- `OMP` (Orthogonal Matching Pursuit) - итеративный алгоритм, который находит разреженное представление сигнала

### Основной класс (CompressiveSensing)

Класс `CompressiveSensing` объединяет все компоненты и предоставляет методы для:
- Сжатия сигнала
- Восстановления сигнала
- Оценки качества восстановления

### Вспомогательные функции

- `generate_test_signal` - генерация различных тестовых сигналов

## Использование библиотеки

### Базовый пример

```python
import numpy as np
from csmp_framework import CompressiveSensing, DCTBasis, MP

# Создание экземпляра CS с ДКП базисом
cs = CompressiveSensing(basis=DCTBasis())

# Сжатие сигнала
compressed_signal = cs.compress(
    signal=original_signal,
    compression_ratio=0.3,  # сжатие до 30% от исходной длины
    sampling_method='random_rows'
)

# Восстановление сигнала с помощью MP
mp_algorithm = MP()
reconstructed_signal = cs.reconstruct(
    compressed_signal=compressed_signal,
    signal_length=len(original_signal),
    algorithm=mp_algorithm,
    max_iter=100,
    epsilon=1e-6
)

# Оценка качества восстановления
metrics = cs.evaluate(original_signal, reconstructed_signal)
```

### Генерация тестового сигнала

```python
from csmp_framework import generate_test_signal

# Синусоидальный сигнал
sinusoid = generate_test_signal(
    signal_type='sinusoid',
    length=256,
    freq1=0.05,
    freq2=0.15,
    freq3=0.3
)

# Разреженный сигнал
sparse = generate_test_signal(
    signal_type='sparse',
    length=256,
    K=10  # количество ненулевых элементов
)

# Ступенчатый сигнал
step = generate_test_signal(
    signal_type='step',
    length=256,
    step_positions=[64, 192]
)

# Частотно-модулированный сигнал (chirp)
chirp = generate_test_signal(
    signal_type='chirp',
    length=256,
    f0=0.01,
    f1=0.4
)
```

## Подробное описание API

### Базисы

#### Абстрактный класс `Basis`

```python
class Basis(ABC):
    @abstractmethod
    def forward(self, signal: np.ndarray) -> np.ndarray:
        """Прямое преобразование сигнала в выбранный базис."""
        
    @abstractmethod
    def backward(self, coefficients: np.ndarray) -> np.ndarray:
        """Обратное преобразование из базиса в сигнал."""
        
    @abstractmethod
    def get_matrix(self, signal_length: int) -> np.ndarray:
        """Получение матрицы базиса для сигнала заданной длины."""
```

#### Класс `DCTBasis`

```python
class DCTBasis(Basis):
    def __init__(self):
        """Инициализация базиса ДКП."""
        
    def forward(self, signal: np.ndarray) -> np.ndarray:
        """Прямое ДКП преобразование."""
        
    def backward(self, coefficients: np.ndarray) -> np.ndarray:
        """Обратное ДКП преобразование."""
        
    def get_matrix(self, signal_length: int) -> np.ndarray:
        """Построение матрицы ДКП для сигнала заданной длины."""
```

#### Класс `DFTBasis`

```python
class DFTBasis(Basis):
    def __init__(self):
        """Инициализация базиса ДПФ."""
        
    def forward(self, signal: np.ndarray) -> np.ndarray:
        """Прямое ДПФ преобразование."""
        
    def backward(self, coefficients: np.ndarray) -> np.ndarray:
        """Обратное ДПФ преобразование."""
        
    def get_matrix(self, signal_length: int) -> np.ndarray:
        """Построение матрицы ДПФ для сигнала заданной длины."""
```

### Матрица выбора отсчетов

```python
class SamplingMatrix:
    @staticmethod
    def random_rows(rows: int, cols: int) -> np.ndarray:
        """Создание матрицы выбора отсчетов со случайными строками."""
        
    @staticmethod
    def gaussian(rows: int, cols: int) -> np.ndarray:
        """Создание матрицы с элементами из нормального распределения."""
        
    @staticmethod
    def bernoulli(rows: int, cols: int) -> np.ndarray:
        """Создание матрицы с элементами из распределения Бернулли."""
```

### Алгоритмы восстановления

#### Абстрактный класс `ReconstructionAlgorithm`

```python
class ReconstructionAlgorithm(ABC):
    @abstractmethod
    def reconstruct(self, sensing_matrix: np.ndarray, compressed_signal: np.ndarray, **kwargs) -> np.ndarray:
        """Восстановление разреженного представления сигнала."""
```

#### Класс `MP` (Matching Pursuit)

```python
class MP(ReconstructionAlgorithm):
    def reconstruct(self, 
                  sensing_matrix: np.ndarray, 
                  compressed_signal: np.ndarray, 
                  max_iter: int = 100,
                  epsilon: float = 1e-6,
                  sparsity: Optional[int] = None) -> np.ndarray:
        """
        Восстановление разреженного представления сигнала методом MP.
        
        Args:
            sensing_matrix: В простом случае (выбор случайных отсчётов сигнала и соответствующих строк из матрицы базиса) - это частичная матрица базиса.
            В более сложных случаях - это матрица выбора отсчётов.
            compressed_signal: Сжатый сигнал y.
            max_iter: Максимальное количество итераций.
            epsilon: Порог ошибки для остановки алгоритма.
            sparsity: Заданная разреженность K (если None, используется epsilon).
            
        Returns:
            Восстановленное разреженное представление сигнала.
        """
```

#### Класс `OMP` (Orthogonal Matching Pursuit)

```python
class MP(ReconstructionAlgorithm):
    def reconstruct(self, 
                  sensing_matrix: np.ndarray, 
                  compressed_signal: np.ndarray, 
                  max_iter: int = 100,
                  epsilon: float = 1e-6,
                  sparsity: Optional[int] = None) -> np.ndarray:
        """
        Восстановление разреженного представления сигнала методом OMP.

        Args:
            sensing_matrix: В простом случае (выбор случайных отсчётов сигнала и соответствующих строк из матрицы базиса) - это частичная матрица базиса.
            В более сложных случаях - это матрица выбора отсчётов.
            compressed_signal: Сжатый сигнал y.
            max_iter: Максимальное количество итераций.
            epsilon: Порог ошибки для остановки алгоритма.
            sparsity: Заданная разреженность K (если None, используется epsilon).

        Returns:
            Восстановленное разреженное представление сигнала.
        """
```

### Основной класс `CompressiveSensing`

```python
class CompressiveSensing:
    def __init__(self, basis: Basis):
        """
        Инициализация CS с выбранным базисом.
        
        Args:
            basis: Базис для представления сигнала.
        """
    
    def compress(self, 
                signal: np.ndarray, 
                compression_ratio: float,
                sampling_method: str = 'random_rows') -> np.ndarray:
        """
        Сжатие сигнала с помощью CS.
        
        Args:
            signal: Входной сигнал.
            compression_ratio: Коэффициент сжатия (0 < ratio < 1).
            sampling_method: Метод выбора отсчетов ('random_rows', 'gaussian', 'bernoulli').
            
        Returns:
            Сжатый сигнал.
        """
    
    def reconstruct(self, 
                  compressed_signal: np.ndarray, 
                  signal_length: int,
                  algorithm: ReconstructionAlgorithm,
                  **kwargs) -> np.ndarray:
        """
        Восстановление сигнала по сжатому представлению.
        
        Args:
            compressed_signal: Сжатый сигнал.
            signal_length: Длина исходного сигнала.
            algorithm: Алгоритм восстановления.
            **kwargs: Дополнительные параметры для алгоритма.
            
        Returns:
            Восстановленный сигнал.
        """
    
    def evaluate(self, 
                original_signal: np.ndarray, 
                reconstructed_signal: np.ndarray) -> Dict[str, float]:
        """
        Оценка качества восстановления сигнала.
        
        Args:
            original_signal: Исходный сигнал.
            reconstructed_signal: Восстановленный сигнал.
            
        Returns:
            Словарь с метриками оценки качества.
        """
```

## Расширение фреймворка

### Добавление нового базиса

Для добавления нового базиса необходимо унаследоваться от абстрактного класса `Basis` и реализовать методы:
- `forward`
- `backward`
- `get_matrix`

```python
class NewBasis(Basis):
    def forward(self, signal: np.ndarray) -> np.ndarray:
        # реализация прямого преобразования
        
    def backward(self, coefficients: np.ndarray) -> np.ndarray:
        # реализация обратного преобразования
        
    def get_matrix(self, signal_length: int) -> np.ndarray:
        # построение матрицы базиса
```

### Добавление нового алгоритма восстановления

Для добавления нового алгоритма восстановления необходимо унаследоваться от абстрактного класса `ReconstructionAlgorithm` и реализовать метод `reconstruct`:

```python
class NewAlgorithm(ReconstructionAlgorithm):
    def reconstruct(self, sensing_matrix: np.ndarray, compressed_signal: np.ndarray, **kwargs) -> np.ndarray:
        # реализация алгоритма восстановления
```
