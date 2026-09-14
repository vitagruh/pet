# 🚀 Инструкция по запуску проекта

## Быстрый старт

### 1. Установка зависимостей

```bash
cd /workspace/pet_reconstruction_analysis

# Вариант A: pip
pip install -r requirements.txt

# Вариант B: conda (рекомендуется)
conda env create -f environment.yml
conda activate pet_reconstruction
```

### 2. Генерация тестового фантома (если нет реальных DICOM)

```bash
python generate_test_phantom.py
```

Это создаст синтетический DICOM-фантом NEMA IEC Body в папке `data/dicom_pet/`.

### 3. Запуск Jupyter Notebook

```bash
jupyter notebook PET_Reconstruction_Analysis.ipynb
```

Notebook автоматически:
- Загрузит DICOM данные из `data/dicom_pet/`
- Выполнит все этапы анализа
- Сгенерирует графики и отчёт

---

## 📁 Структура проекта

```
pet_reconstruction_analysis/
├── generate_test_phantom.py    # Скрипт генерации тестового фантома
├── PET_Reconstruction_Analysis.ipynb  # Главный notebook
├── README.md                    # Документация
├── requirements.txt             # Python зависимости
├── environment.yml              # Conda окружение
│
├── src/                         # Модули
│   ├── dicom_loader.py          # Загрузка DICOM, расчёт SUV
│   ├── phantom_generator.py     # Генератор фантома NEMA IEC
│   ├── reconstruction_sim.py    # Симуляция OSEM реконструкции
│   ├── metrics.py               # Расчёт метрик (RC, SUV, %CV)
│   └── visualization.py         # Графики и отчёты
│
├── data/
│   └── dicom_pet/               # Входные DICOM файлы
│
└── results/                     # Результаты анализа
    ├── processed/               # Промежуточные данные
    ├── figures/                 # Графики
    └── reports/                 # Отчёты
```

---

## 🔬 Что делает пайплайн

### Этап 1: Загрузка DICOM
- Рекурсивный поиск `.dcm` файлов
- Сортировка по SliceLocation
- Извлечение калибровочных коэффициентов
- Конвертация в SUV с учётом распада

### Этап 2: Варьирование параметров
- **Итерации**: [2, 3, 4, 6]
- **Подмножества**: [16, 21, 28]
- **Матрица**: [128, 192, 256]

### Этап 3: Симуляция реконструкции
Научно обоснованная аппроксимация OSEM без sinogram:
- Модель сходимости по Laforest et al., JNM 2016
- Итеративное улучшение резкости + шум

### Этап 4: Расчёт метрик
Для каждой сферы фантома:
- `SUVmax`, `SUVmean`, `SUVpeak`
- `Recovery Coefficient (RC)`
- `Noise (%CV)`

### Этап 5: Визуализация
- Heatmap: RC vs (iterations × subsets)
- Line plots: SUVpeak vs iterations
- Boxplot: распределение %CV
- Markdown-отчёт с рекомендациями

---

## ⚙️ Конфигурация

Откройте `PET_Reconstruction_Analysis.ipynb` и измените `CONFIG` в начале:

```python
CONFIG = {
    'dicom_path': 'data/dicom_pet',
    'output_dir': 'results',
    'iterations': [2, 3, 4, 6],
    'subsets': [16, 21, 28],
    'matrices': [128, 192, 256],
    'random_seed': 42
}
```

---

## 📊 Ожидаемые результаты

После выполнения notebook вы получите:

1. **Графики** в `results/figures/`:
   - `rc_heatmap_matrix_*.png` - heatmap Recovery Coefficient
   - `suvpeak_vs_iterations.png` - SUVpeak от итераций
   - `noise_boxplot.png` - распределение шума

2. **Данные** в `results/processed/`:
   - `metrics_summary.csv` - сводная таблица метрик
   - `reconstructions/` - обработанные volumes

3. **Отчёт** в `results/reports/`:
   - `analysis_report.md` - Markdown с интерпретацией

---

## ⚠️ Важные примечания

### О симуляции реконструкции

Полная OSEM-реконструкция требует raw sinogram-данных, которые недоступны без vendor-specific информации. 

В проекте реализована **научно обоснованная аппроксимация**:
- Модель роста разрешения и шума
- Калибрована по литературным данным (Laforest et al., JNM 2016; NEMA NU-2-2018)
- Допустима для курсовой работы и образовательных целей

### Для реальных клинических данных

Замените содержимое `data/dicom_pet/` на ваши DICOM-файлы ПЭТ с:
- Коррекцией затухания
- Калибровкой активности
- Метаданными (вес пациента, введённая доза, время инъекции)

---

## 🐛 Решение проблем

### Ошибка: "No PET DICOM files found"
Проверьте, что файлы имеют тег `Modality = PT` или `PET`.

### Ошибка: "Invalid DICOM file"
Убедитесь, что файлы не повреждены и содержат полные метаданные.

### Ошибка импорта модулей
```bash
export PYTHONPATH=/workspace/pet_reconstruction_analysis/src:$PYTHONPATH
```

---

## 📚 Литература

1. Laforest R. et al. "Evaluation of quantitative accuracy in PET imaging." JNM 2016.
2. NEMA NU-2-2018. "Performance Measurements of Positron Emission Tomographs."
3. EANM Guidelines on FDG PET/CT. Eur J Nucl Med Mol Imaging 2015.

---

## ✅ Проверка работоспособности

```bash
cd /workspace/pet_reconstruction_analysis
python -c "
import sys
sys.path.insert(0, 'src')
from dicom_loader import load_pet_dicom
pet = load_pet_dicom('data/dicom_pet')
print(f'✅ Загружено: {pet.image_array.shape}, SUV range: [{pet.image_array.min():.3f}, {pet.image_array.max():.3f}]')
"
```

Ожидаемый вывод:
```
✅ Загружено: (80, 192, 192), SUV range: [0.xxx, x.xxx]
```

---

## 🎯 Следующие шаги

1. Запустите `generate_test_phantom.py` для создания тестовых данных
2. Откройте `PET_Reconstruction_Analysis.ipynb` в Jupyter
3. Выполните все ячейки последовательно
4. Изучите результаты в папке `results/`
5. При необходимости настройте параметры в CONFIG

**Приятной работы!** 🚀
