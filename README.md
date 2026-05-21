# VoiceClone-GUI-portable — запуск из исходного кода / Running from source

Это руководство предназначено для запуска программы **VoiceClone-GUI** непосредственно из исходного Python-кода на другом компьютере. Программа выполняет клонирование голоса (XTTS v2), распознавание речи (Whisper) и **офлайн-перевод** распознанного или вводимого текста (Argos Translate) полностью локально, сохраняя данные в папку программы.

This guide explains how to run **VoiceClone-GUI** directly from the Python source code on another computer. The program performs voice cloning (XTTS v2), speech recognition (Whisper) and **offline translation** of recognized or input text (Argos Translate) entirely locally, saving all data in program's folder.

---

## Системные требования / System Requirements

- **ОС / OS**: Windows 10/11 (64-bit) — рекомендуется; возможна работа на Linux/macOS с незначительными изменениями.
- **Python**: версия 3.10–3.12 (Python 3.13 и выше не поддерживаются некоторыми библиотеками). Рекомендуется 3.12.6.
- **Процессор / CPU**: Intel Core i5 или аналогичный (многоядерный рекомендуется) / Intel Core i5 or equivalent (multi-core recommended)
- **Оперативная память / RAM**: минимум 8 ГБ (рекомендуется 16 ГБ) / minimum 8 GB (16 GB recommended)
- **Диск / Storage**: ~12 ГБ свободного места для моделей и кэша / ~12 GB free space for models and cache
- **Дополнительно / Additional**: **FFmpeg** (полная shared-сборка / full shared build) – необходима для работы аудио-обработки и torchcodec.

---

## 1. Установка Python / Installing Python

Скачайте и установите Python 3.12.6 с официального сайта: [python.org](https://www.python.org/downloads/release/python-3126/). При установке обязательно отметьте галочку **«Add Python to PATH»**.

Download and install Python 3.12.6 from the official website: [python.org](https://www.python.org/downloads/release/python-3126/). During installation, make sure to check **"Add Python to PATH"**.

Проверьте установку, открыв терминал (cmd) и выполнив:
```bash
python --version
```
Должно отобразиться `Python 3.12.6`.

Verify the installation by opening a terminal (cmd) and running:
```bash
python --version
```
It should display `Python 3.12.6`.

---

## 2. Получение кода / Getting the Code

Скопируйте все файлы проекта (включая `main.py`) в отдельную папку, например `C:\VoiceClone-GUI_source`. Убедитесь, что у вас есть файл `main.py` (основной скрипт) и, возможно, другие вспомогательные файлы (иконка и т.д.).

Copy all project files (including `main.py`) into a separate folder, e.g., `C:\VoiceClone-GUI_source`. Ensure you have the `main.py` file (the main script) and possibly other auxiliary files (icon, etc.).

---

## 3. Создание виртуального окружения (рекомендуется) / Creating a Virtual Environment (recommended)

Откройте терминал в папке проекта и выполните:

Open a terminal in the project folder and run:

```bash
python -m venv venv
```

Активируйте окружение / Activate the environment:

- **Windows (cmd)**:
  ```bash
  venv\Scripts\activate
  ```
- **Windows (PowerShell)**:
  ```bash
  venv\Scripts\Activate.ps1
  ```
- **Linux/macOS**:
  ```bash
  source venv/bin/activate
  ```

После активации в начале строки терминала появится `(venv)`.

After activation, you should see `(venv)` at the beginning of the terminal prompt.

---

## 4. Установка зависимостей / Installing Dependencies

Убедитесь, что pip обновлён / Ensure pip is up to date:

```bash
python -m pip install --upgrade pip
```

Установите необходимые библиотеки. Важно: для совместимости с torchcodec и TTS рекомендуется использовать PyTorch версии 2.8.0 (или более новую, но с осторожностью). Ниже приведён список пакетов, которые нужно установить:

Install the required libraries. Important: for compatibility with torchcodec and TTS, it is recommended to use PyTorch version 2.8.0 (or newer with caution). The list of packages to install is below:

```bash
# CPU version (recommended for compatibility)
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
# GPU version (if you have CUDA 12.4)
# pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu124

pip install openai-whisper
pip install TTS
pip install argostranslate
pip install sounddevice soundfile pygame webrtcvad
pip install numpy scipy matplotlib
pip install transformers huggingface_hub tokenizers safetensors
pip install gruut ko_speech_tools
pip install inflect typeguard
pip install fsspec
pip install pysbd
pip install librosa
pip install scikit-learn
```

Примечание: torchcodec устанавливать не нужно – программа использует обходной путь (PyTorch 2.8.0) и не требует его.

Note: torchcodec does not need to be installed – the program uses a workaround (PyTorch 2.8.0) and does not require it.

Если вы хотите точное воспроизведение версий, используйте файл `requirements.txt` со следующим содержимым (пример):

If you need exact version reproduction, use a `requirements.txt` file with the following content (example):

```
torch==2.8.0
torchaudio==2.8.0
openai-whisper==20231117
TTS==0.22.0
argostranslate==1.9.0
sounddevice==0.5.1
soundfile==0.13.1
pygame==2.6.1
webrtcvad==2.0.10
numpy==2.1.1
scipy==1.17.0
matplotlib==3.10.1
transformers==4.36.0
huggingface_hub==0.36.2
tokenizers==0.15.2
safetensors==0.7.0
gruut==2.3.2
ko_speech_tools==0.2.0
inflect==7.5.0
typeguard==4.4.2
fsspec==2025.3.0
pysbd==0.3.4
librosa==0.11.0
scikit-learn==1.6.1
```

Затем выполните:

Then run:

```bash
pip install -r requirements.txt
```

---

## 5. Установка FFmpeg / Installing FFmpeg

Программа требует наличия полной shared-сборки FFmpeg (с DLL) для работы аудио-операций.

The program requires the full shared build of FFmpeg (with DLLs) for audio operations.

### Windows

1. Скачайте **полную shared-сборку FFmpeg** с официального сайта:  
   [https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/)  
   Выберите **ffmpeg-release-full-shared.7z**.
2. Распакуйте архив. Из папки `bin` скопируйте **все файлы** (в том числе `avcodec-*.dll`, `avformat-*.dll`, `avutil-*.dll`, `swresample-*.dll` и др.) в папку `ffmpeg`, созданную **рядом с вашим скриптом `main.py`**. Если папки `ffmpeg` нет – создайте её вручную.
3. Убедитесь, что в папке `ffmpeg` есть исполняемый файл `ffmpeg.exe` и все необходимые DLL.

Альтернативно, можно добавить путь к `bin` FFmpeg в системную переменную `PATH`, но программа сама добавит локальную папку `ffmpeg` в `PATH` при запуске (как указано в коде).

### Linux / macOS

Установите FFmpeg через пакетный менеджер (например, `sudo apt install ffmpeg` для Ubuntu, `brew install ffmpeg` для macOS). Убедитесь, что команда `ffmpeg` доступна в терминале.

---

## 6. Запуск программы / Running the Program

Убедитесь, что виртуальное окружение активировано. Затем выполните:

Make sure the virtual environment is activated. Then run:

```bash
python main.py
```

При первом запуске программа создаст рядом со скриптом папки:
- `cache` – для хранения моделей и кэша.
- `input` – для входных файлов.
- `output` – для результатов.

При первом использовании клонирования голоса будет автоматически загружена модель XTTS v2 (~1.8 ГБ). При первом распознавании – модель Whisper (размер зависит от выбора). Для перевода потребуется однократно загрузить языковые пакеты Argos Translate (по запросу). Для загрузки требуется интернет.

On first run, the program will create the following folders next to the script:
- `cache` – for models and cache.
- `input` – for input files.
- `output` – for results.

The first time you use voice cloning, the XTTS v2 model (~1.8 GB) will be downloaded automatically. The first time you perform recognition, the Whisper model (size depends on selection) will be downloaded. For translation, you will need to download language packages via Argos Translate (on request). Internet connection is required for downloads.

---

## 7. Функции перевода / Translation features

Программа поддерживает офлайн-перевод распознанного текста на 17 языков, а также перевод вводимого текста перед синтезом. Поддерживаемые языки: русский, английский, испанский, французский, немецкий, итальянский, португальский, польский, турецкий, китайский, японский, корейский, голландский, чешский, арабский, венгерский, хинди.

При выборе языка перевода, отличного от исходного, программа проверит наличие соответствующего пакета Argos Translate. Если пакет отсутствует, будет предложено его скачать (однократно).

The program supports offline translation of recognized text into 17 languages, as well as translation of input text before synthesis. Supported languages: Russian, English, Spanish, French, German, Italian, Portuguese, Polish, Turkish, Chinese, Japanese, Korean, Dutch, Czech, Arabic, Hungarian, Hindi.

When you select a target language different from the source, the program will check for the corresponding Argos Translate package. If the package is missing, you will be prompted to download it (once).

---

## 8. Возможные проблемы и их решение / Troubleshooting

### 8.1. Ошибка «Could not load libtorchcodec» или «FFmpeg not found»
- Убедитесь, что в папке `ffmpeg` рядом со скриптом находятся все DLL из полной shared-сборки FFmpeg.
- Проверьте, что в этой папке есть файлы `avcodec-*.dll`, `avformat-*.dll` и т.д.
- Если вы не хотите использовать локальную папку, добавьте путь к FFmpeg в системную переменную `PATH` и перезапустите терминал.
- Убедитесь, что используется PyTorch 2.8.0 (или более старая версия, совместимая с torchcodec). Программа рассчитана на PyTorch 2.8.0, который не требует torchcodec

### 8.2. Ошибка «No module named '...'»
- Эта ошибка возникает из-за несовместимости патча с новой версией transformers. В коде main.py уже есть исправленный патч. Убедитесь, что вы используете актуальную версию скрипта (с поддержкой именованных аргументов).

### 8.3. Ошибка «No module named '...'»
- Убедитесь, что виртуальное окружение активировано.
- Проверьте, что все зависимости установлены (запустите `pip list` и сравните с требуемыми).
- Попробуйте переустановить проблемный пакет: `pip install --upgrade <package>`.

### 8.4. Ошибка «[Errno 2] No such file or directory: '.../mel_filters.npz'»
- Эта ошибка возникает, если whisper не может найти свои файлы данных. Обычно они загружаются автоматически при первом импорте. Попробуйте удалить папку `cache` и запустить программу заново – файлы должны скачаться.

### 8.5. Предупреждения от `inflect` и `typeguard`
- Предупреждения вида `InstrumentationWarning` безопасны и не влияют на работу. Они связаны с декораторами проверки типов. Чтобы их скрыть, можно установить переменную окружения `TYPEGUARD_DISABLE=1` (код уже делает это для собранного EXE, но для исходного кода можно добавить вручную или игнорировать).

### 8.6. Долгая загрузка при первом запуске
- Нормально – модели скачиваются из интернета. При повторных запусках они будут использоваться локально.

---

## Лицензии и благодарности / Licenses and Acknowledgements

Программа использует:
- [Coqui TTS](https://github.com/coqui-ai/TTS) (XTTS v2)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Argos Translate] (https://github.com/argosopentech/argos-translate)
- [PyTorch](https://pytorch.org/)
- [FFmpeg](https://ffmpeg.org/)

Все компоненты распространяются под своими лицензиями (MIT, Apache 2.0, GPL и др.). Данный код предназначен для запуска в исследовательских и личных целях.

The software uses:
- [Coqui TTS](https://github.com/coqui-ai/TTS) (XTTS v2)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Argos Translate] (https://github.com/argosopentech/argos-translate)
- [PyTorch](https://pytorch.org/)
- [FFmpeg](https://ffmpeg.org/)

All components are distributed under their respective licenses (MIT, Apache 2.0, GPL, etc.). This code is intended for research and personal use.

---
