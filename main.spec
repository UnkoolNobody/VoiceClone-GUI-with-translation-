# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, copy_metadata

# Увеличиваем лимит рекурсии
sys.setrecursionlimit(100000)

# Отключаем автоматическую компиляцию glib schema, вызывавшую ошибку
import PyInstaller.utils.hooks.gi as gi_hook
def noop(*args, **kwargs):
    return []
gi_hook.compile_glib_schema_files = noop

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('locales', 'locales'),                       # папка с файлами локализации
        ('input', 'input'),                           # папка для входных файлов
        ('output', 'output'),                         # папка для выходных файлов
        ('input/reference_samples', 'input/reference_samples'),  # образцы голоса
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib', 'IPython', 'jedi', 'parso', 'zmq',
        'pandas', 'tensorflow', 'tensorflow.*', 'keras', 'jax', 'flax',
        'notebook', 'jupyter_client', 'jupyter_core', 'ipykernel',
        'PyQt5', 'PySide2', 'PyQt6', 'PySide6',
        'ipywidgets', 'qtpy', 'qtconsole',
        'torchvision', 'tensorboard', 'tensorboardX',
        'numba', 'llvmlite',
    ],
    noarchive=False,
)

# ========== СБОР ВСЕХ ПОДМОДУЛЕЙ ==========
a.hiddenimports += collect_submodules('TTS')
a.hiddenimports += collect_submodules('whisper')
a.hiddenimports += collect_submodules('argostranslate')
a.hiddenimports += collect_submodules('transformers')
a.hiddenimports += collect_submodules('transformers.models')
a.hiddenimports += collect_submodules('torchaudio')
a.hiddenimports += collect_submodules('sounddevice')
a.hiddenimports += collect_submodules('pygame')
a.hiddenimports += collect_submodules('scipy')
a.hiddenimports += collect_submodules('numpy')
a.hiddenimports += collect_submodules('webrtcvad')

# Дополнительные скрытые импорты, обнаруженные в коде
a.hiddenimports += [
    'scipy._lib._ccallback_c',
    'scipy._cyutility',
    'array_api_compat',
    'array_api_compat.numpy',
    'torch._C',
    'torch._C._cudart',
    'torchaudio._torchaudio',
    'torchaudio.backend',
    '_sounddevice_data',
    'pkg_resources',
    'importlib.metadata',
]

# ========== ДОБАВЛЕНИЕ DATA-ФАЙЛОВ ==========
for pkg in ['TTS', 'argostranslate', 'transformers', 'whisper']:
    a.datas += collect_data_files(pkg)

# Копируем метаданные пакетов
for pkg in ['tqdm', 'regex', 'tokenizers', 'safetensors', 'packaging',
            'requests', 'filelock', 'pyyaml', 'numpy', 'argostranslate']:
    try:
        a.datas += copy_metadata(pkg)
    except:
        pass

# Фильтруем некорректные записи в a.binaries и a.datas (исправляет ошибку "not enough values to unpack")
a.binaries = [entry for entry in a.binaries if len(entry) == 3]
a.datas = [entry for entry in a.datas if len(entry) == 3]

# ========== СБОРКА PYZ ==========
pyz = PYZ(a.pure)

# ========== СБОРКА ИСПОЛНЯЕМОГО ФАЙЛА ==========
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
    upx_exclude=['*.dll', '*.so', '*.dylib'],   # исключаем бинарные библиотеки из сжатия UPX
    runtime_tmpdir=None,
    console=False,           # скрыть консольное окно (GUI)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico' if os.path.exists('icon.ico') else None,
)
