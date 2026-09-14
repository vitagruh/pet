# Влияние параметров реконструкции ПЭТ на точность восстановления удельной активности

## 📋 Описание проекта

Данный проект представляет собой воспроизводимый пайплайн для анализа влияния параметров итеративной реконструкции OSEM на количественную точность ПЭТ-визуализации.

### Цели исследования:
- Оценка влияния **количества итераций** на Recovery Coefficient
- Оценка влияния **числа подмножеств** на уровень шума
- Оценка влияния **размера матрицы** на пространственное разрешение
- Генерация сравнительных графиков и отчёта

---

## 📁 Структура проекта

```
pet_reconstruction_analysis/
├── src/                          # Исходный код модулей
│   ├── dicom_loader.py           # Загрузка DICOM, расчёт SUV
│   ├── phantom_generator.py      # Генератор фантома NEMA IEC
│   ├── reconstruction_sim.py     # Симуляция OSEM реконструкции
│   ├── metrics.py                # Расчёт метрик (RC, SUV, %CV)
│   └── visualization.py          # Графики и генерация отчётов
├── data/
│   └── dicom_pet/                # Входные DICOM-данные
├── results/
│   ├── processed/                # Промежуточные результаты
│   ├── figures/                  # Графики
│   └── reports/                  # Отчёты
├── PET_Reconstruction_Analysis.ipynb  # Главный Jupyter Notebook
├── requirements.txt              # Зависимости Python
├── environment.yml               # Conda окружение
└── README.md                     # Этот файл
```

---

## ⚙️ Установка

### Вариант 1: Conda (рекомендуется)

```bash
cd pet_reconstruction_analysis
conda env create -f environment.yml
conda activate pet_reconstruction
```

### Вариант 2: pip

```bash
cd pet_reconstruction_analysis
pip install -r requirements.txt
```

---

## 🚀 Быстрый старт

### Запуск с синтетическим фантомом (по умолчанию)

```bash
jupyter notebook PET_Reconstruction_Analysis.ipynb
```

Затем выполнить все ячейки sequentially (Cell → Run All).

### Запуск через Python скрипт

```python
# generate_and_analyze.py
from src.phantom_generator import generate_nema_phantom
from src.reconstruction_sim import ReconstructionSimulator
from src.metrics import MetricsCalculator
from src.visualization import PETVisualizer, ReportGenerator

# 1. Генерация фантома
volume, metadata = generate_nema_phantom(
    output_dir='./data/dicom_pet',
    matrix_size=192,
    sphere_ratio=4.0,
    noise_level=0.1,
    seed=42
)

# 2. Реконструкция и анализ (см. Notebook)
```

---

## 📊 Ожидаемые результаты

После выполнения Notebook будут созданы:

### Графики (`results/figures/`):
- `rc_heatmap.png` — Heatmap Recovery Coefficient vs параметры
- `suvpeak_vs_iter.png` — SUVpeak vs итерации для разных матриц
- `noise_boxplot.png` — Распределение %CV по подмножествам
- `rc_vs_sphere_size.png` — RC в зависимости от размера сферы
- `comparison_slices.png` — Сравнительные срезы

### Данные (`results/processed/`):
- `all_metrics.csv` — Все измеренные метрики
- `background_metrics.csv` — Метрики фона
- `summary_statistics.csv` — Сводная статистика
- `reconstructions.pkl` — Все реконструированные объёмы

### Отчёт (`results/reports/`):
- `analysis_report.md` — Полный Markdown-отчёт с интерпретацией

---

## 🔬 Научное обоснование

### Аппроксимация OSEM реконструкции

**Важное замечание:** Полная OSEM-реконструкция требует raw sinogram-данных, которые обычно недоступны из клинических DICOM.

Для целей данной работы реализована **научно обоснованная аппроксимация**:

1. **Модель сходимости** [Laforest et al., JNM 2016]:
   ```
   RC(iter) = RC_max × (1 - exp(-k × iter × subsets / 21))
   ```

2. **Модель шума**:
   ```
   Noise(iter) = Noise₀ × √(1 + β × iter)
   ```

3. **Пост-фильтрация** зависит от числа подмножеств

Это допустимо для курсовой работы, так как:
- Воспроизводит основные тенденции реальных данных
- Позволяет изучить относительное влияние параметров
- Соответствует литературным данным

---

## 📐 Физические формулы

### SUV (Standardized Uptake Value)
```
SUV = (Activity_voxel [kBq/ml] × PatientWeight [kg]) / InjectedDose [kBq]
```

### Recovery Coefficient
```
RC = Measured_SUV / True_SUV
```

### Коэффициент вариации (шум)
```
%CV = 100 × σ_background / μ_background
```

---

## 📚 Литература

1. Laforest R, et al. "Evaluation of Quantitative Accuracy in PET/CT." J Nucl Med. 2016
2. NEMA NU-2-2018. "Performance Measurements of Positron Emission Tomographs."
3. EANM Research Ltd. (EARL) Accreditation Standards for PET/CT
4. Boellaard R, et al. "FDG PET and PET/CT: EANM procedure guidelines for tumour PET imaging." Eur J Nucl Med Mol Imaging. 2015;42:328-354.

---

## 💡 Рекомендации по использованию

### Для количественного анализа (SUV измерения):
- **Итерации:** 3-4
- **Подмножества:** 21
- **Матрица:** 192×192

### Для визуальной оценки (обнаружение очагов):
- **Итерации:** 4-6
- **Подмножества:** 21-28
- **Матрица:** 256×256

---

## 🛠 Расширение функциональности

### Добавление реальных DICOM-данных

1. Поместите DICOM-файлы в `./data/dicom_pet/`
2. В Notebook установите `USE_SYNTHETIC_PHANTOM = False`
3. Обеспечьте наличие метаданных (вес, доза, время инъекции)

### Интеграция с STIR/ASTRA

Для полноценной OSEM реконструкции:

```python
# Пример интеграции (требует установки STIR)
from stir import OSEMReconstructor

reconstructor = OSEMReconstructor(
    sinogram=sinogram_data,
    iterations=n_iter,
    subsets=n_subsets
)
reconstructed = reconstructor.run()
```

---

## 📄 Лицензия

Проект создан в образовательных целях. Используйте согласно академической добросовестности.

---

## 👨‍💻 Автор

Курсовая работа по медицинской визуализации, 2024
