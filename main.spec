# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, copy_metadata

# Увеличиваем лимит рекурсии (на случай больших графов импорта)
sys.setrecursionlimit(10000)

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Исключаем ненужные большие модули, которые не используются в коде
        'matplotlib', 'IPython', 'jedi', 'parso', 'zmq',
        'pandas', 'tensorflow', 'tensorflow.*', 'keras', 'jax', 'flax',
        'notebook', 'jupyter_client', 'jupyter_core', 'ipykernel',
        'PyQt5', 'PySide2', 'PyQt6', 'PySide6',
        'ipywidgets', 'qtpy', 'qtconsole',
        'torchvision',
    ],
    noarchive=False,
)

# Добавляем все подмодули для TTS, Whisper и Argos Translate
a.hiddenimports += collect_submodules('TTS')
a.hiddenimports += collect_submodules('whisper')
a.hiddenimports += collect_submodules('argostranslate')
a.hiddenimports += collect_submodules('transformers')
a.hiddenimports += ['scipy._lib._ccallback_c', 'scipy._cyutility', 'array_api_compat', 'array_api_compat.numpy']

# Добавляем data-файлы (конфиги, языковые ресурсы)
a.datas += collect_data_files('TTS')
a.datas += collect_data_files('argostranslate')
a.datas += collect_data_files('transformers')
a.datas += collect_data_files('whisper')

# Копируем метаданные для пакетов, которые требуют проверки версий
for pkg in ['tqdm', 'regex', 'tokenizers', 'safetensors', 'packaging', 'requests', 'filelock', 'pyyaml', 'numpy']:
    try:
        a.datas += copy_metadata(pkg)
    except:
        pass

# Ручное добавление скрытых импортов из кода
a.hiddenimports += [
    'webrtcvad',
    'sounddevice',
    'pygame',
    'soundfile',
    'torchaudio',
    'torch',
    'pkg_resources',
    'importlib.metadata',
    'coqui_tts',
]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='VoiceClone',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,        # скрываем консоль
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico' if os.path.exists('icon.ico') else None,
)
