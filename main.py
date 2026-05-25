# ============================================================
# VoiceClone-GUI - speech recognition, synthesis and voice cloning
# VoiceClone-GUI - распознавание речи, синтез и клонирование голоса
# ============================================================
# Russian / Русский:
# Программа для распознавания речи (Whisper) и синтеза речи
# с клонированием голоса (XTTS v2) + офлайн-перевод (Argos Translate).
# Все модели и кэш хранятся в папке cache рядом с программой.
# FFmpeg должен лежать в папке ffmpeg рядом с программой.
# Добавлен неслышимый цифровой водяной знак (LSB) в выходные аудиофайлы.
#
# English / Английский:
# Speech recognition (Whisper) and speech synthesis with voice
# cloning (XTTS v2) + offline translation (Argos Translate).
# All models and cache are stored in the "cache" folder next to the program.
# FFmpeg must be located in the "ffmpeg" folder next to the program.
# Inaudible digital watermark (LSB) is embedded into output audio files.
# ============================================================

import sys
import os
import locale
import json
import warnings
import subprocess
import traceback
import threading
import time
import tempfile
import numpy as np
import sounddevice as sd
import wave
import pygame
import torch
import torchaudio
import whisper
import webrtcvad
import soundfile as sf
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

# ========== ПАТЧ ДЛЯ TRANSFORMERS / PATCH FOR TRANSFORMERS ==========
import transformers.pytorch_utils
if not hasattr(transformers.pytorch_utils, 'isin_mps_friendly'):
    transformers.pytorch_utils.isin_mps_friendly = torch.isin
    print("✓ Added isin_mps_friendly stub")

# ========== ИМПОРТ TTS / TTS IMPORT ==========
from TTS.api import TTS

# ========== НАСТРОЙКИ ПОРТАТИВНОСТИ / PORTABLE SETTINGS ==========
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ========== ПРИНУДИТЕЛЬНОЕ ПЕРЕНАПРАВЛЕНИЕ ARGOS TRANSLATE ==========
# / FORCED REDIRECTION OF ARGOS TRANSLATE ==========
# Argos Translate по умолчанию складывает всё в ~/.local/share/argos-translate
# и ~/.cache/argos-translate. Мы принудительно перенаправляем в папку программы.
# Argos Translate by default stores data in ~/.local/share/argos-translate
# and ~/.cache/argos-translate. We forcibly redirect to the program's folder.

ARGOS_CACHE_DIR = os.path.join(CACHE_DIR, "argos-translate")
os.makedirs(ARGOS_CACHE_DIR, exist_ok=True)

# Патчим appdirs и platformdirs, чтобы они возвращали наши пути
# Patch appdirs and platformdirs to return our paths
try:
    import appdirs
    _orig_user_cache = appdirs.user_cache_dir
    _orig_user_data = appdirs.user_data_dir
    def _patched_user_cache(appname=None, *args, **kwargs):
        if appname == "argos-translate" or (appname and "argos" in appname.lower()):
            return ARGOS_CACHE_DIR
        return _orig_user_cache(appname, *args, **kwargs)
    def _patched_user_data(appname=None, *args, **kwargs):
        if appname == "argos-translate" or (appname and "argos" in appname.lower()):
            return ARGOS_CACHE_DIR
        return _orig_user_data(appname, *args, **kwargs)
    appdirs.user_cache_dir = _patched_user_cache
    appdirs.user_data_dir = _patched_user_data
    print("✓ Patched appdirs for Argos")
except ImportError:
    pass

try:
    import platformdirs
    _orig_plat_cache = platformdirs.user_cache_dir
    _orig_plat_data = platformdirs.user_data_dir
    def _patched_plat_cache(appname=None, *args, **kwargs):
        if appname == "argos-translate" or (appname and "argos" in appname.lower()):
            return ARGOS_CACHE_DIR
        return _orig_plat_cache(appname, *args, **kwargs)
    def _patched_plat_data(appname=None, *args, **kwargs):
        if appname == "argos-translate" or (appname and "argos" in appname.lower()):
            return ARGOS_CACHE_DIR
        return _orig_plat_data(appname, *args, **kwargs)
    platformdirs.user_cache_dir = _patched_plat_cache
    platformdirs.user_data_dir = _patched_plat_data
    print("✓ Patched platformdirs for Argos")
except ImportError:
    pass

# Дополнительно устанавливаем переменные окружения для Argos
# Also set environment variables for Argos
os.environ['ARGOS_TRANSLATE_PACKAGE_DIRS'] = ARGOS_CACHE_DIR
os.environ['XDG_CACHE_HOME'] = CACHE_DIR
os.environ['XDG_DATA_HOME'] = CACHE_DIR

# Импортируем Argos Translate
# Import Argos Translate
import argostranslate.package
import argostranslate.translate

# Принудительно устанавливаем пути внутри библиотеки
# Forcibly set paths inside the library
argostranslate.package.package_data_dir = ARGOS_CACHE_DIR
argostranslate.package.package_dirs = [ARGOS_CACHE_DIR]
argostranslate.package.downloads_dir = ARGOS_CACHE_DIR   # временные файлы тоже туда / temporary files also there

print(f"Argos Translate will use: {ARGOS_CACHE_DIR}")

# ========== ОСТАЛЬНЫЕ КЭШИ / OTHER CACHES ==========
os.environ['HF_HOME'] = os.path.join(CACHE_DIR, "hf_cache")
os.environ['TRANSFORMERS_CACHE'] = os.path.join(CACHE_DIR, "hf_cache")
os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(CACHE_DIR, "hf_cache")
os.environ['TTS_HOME'] = os.path.join(CACHE_DIR, "tts_cache")

INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
REF_SAMPLES_DIR = os.path.join(INPUT_DIR, "reference_samples")
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REF_SAMPLES_DIR, exist_ok=True)

FFMPEG_DIR = os.path.join(BASE_DIR, "ffmpeg")
if os.path.exists(FFMPEG_DIR):
    os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")
    print(f"Local FFmpeg added to PATH: {FFMPEG_DIR}")

SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")

# ========== ПАТЧ ДЛЯ TRANSFORMERS (полный) / FULL TRANSFORMERS PATCH ==========
if not hasattr(transformers.pytorch_utils, 'isin_mps_friendly'):
    def _isin_mps_friendly(*args, **kwargs):
        a = None
        b = None
        if len(args) >= 2:
            a, b = args[0], args[1]
        elif 'elements' in kwargs and 'test_elements' in kwargs:
            a = kwargs['elements']
            b = kwargs['test_elements']
        elif len(args) == 1 and 'test_elements' in kwargs:
            a = args[0]
            b = kwargs['test_elements']
        elif 'elements' in kwargs and len(args) == 1:
            a = kwargs['elements']
            b = args[0]
        else:
            raise TypeError(f"Cannot parse arguments: args={args}, kwargs={kwargs}")
        return torch.isin(a, b)
    transformers.pytorch_utils.isin_mps_friendly = _isin_mps_friendly
    print("✓ Patch for transformers.isin_mps_friendly applied")

os.environ['COQUI_TOS_AGREED'] = '1'
warnings.filterwarnings('ignore')
pygame.mixer.init()

# ========== ЛОКАЛИЗАЦИЯ / LOCALIZATION ==========
class Localization:
    _instance = None
    _strings = {}
    _current_lang = 'en'

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_available_languages(self):
        locales_dir = os.path.join(BASE_DIR, "locales")
        if not os.path.isdir(locales_dir):
            return ['en']
        files = [f[:-5] for f in os.listdir(locales_dir) if f.endswith('.json')]
        return files if files else ['en']

    def load_language(self, lang_code=None):
        locales_dir = os.path.join(BASE_DIR, "locales")
        os.makedirs(locales_dir, exist_ok=True)

        if lang_code is None:
            try:
                if sys.platform == 'win32':
                    import ctypes
                    lang_id = ctypes.windll.kernel32.GetUserDefaultUILanguage()
                    lang_map = {1033: 'en', 1049: 'ru', 1031: 'de', 1034: 'es', 1036: 'fr', 2052: 'zh'}
                    lang_code = lang_map.get(lang_id, 'en')
                else:
                    lc, _ = locale.getdefaultlocale()
                    lang_code = lc[:2] if lc else 'en'
            except:
                lang_code = 'en'
            lang_code = lang_code.lower()

        json_path = os.path.join(locales_dir, f"{lang_code}.json")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self._strings = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            json_path = os.path.join(locales_dir, "en.json")
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    self._strings = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                self._create_default_en(locales_dir)
                with open(json_path, 'r', encoding='utf-8') as f:
                    self._strings = json.load(f)
        self._current_lang = lang_code
        return self._strings

    def get(self, key, default=None):
        return self._strings.get(key, default or key)

    def _create_default_en(self, locales_dir):
        default_en = {
            "app_title": "VoiceClone-GUI: speech recognition and synthesis with translation",
            "left_panel_title": "Speech Recognition",
            "mode_file": "Select file",
            "mode_record": "Record file",
            "file_label": "File:",
            "browse_button": "Browse",
            "record_label": "Recording:",
            "record_button_start": "Record",
            "record_button_stop": "Stop",
            "output_file_label": "Output file path:",
            "whisper_model_frame": "Whisper model",
            "recognize_button": "Recognize",
            "recognize_translate_button": "Recognize & Translate",
            "stt_status_ready": "Ready",
            "stt_status_loading_model": "Loading {} model...",
            "stt_status_recognizing": "Recognizing...",
            "stt_status_translating": "Translating...",
            "stt_status_checking_package": "Checking translation package...",
            "stt_status_downloading_package": "Downloading package...",
            "stt_status_error": "Error",
            "original_text_frame": "Recognized text",
            "translated_text_frame": "Translation",
            "right_panel_title": "Speech Synthesis",
            "synth_mode_file": "Select file",
            "synth_mode_text": "Enter text",
            "text_file_label": "Text file:",
            "text_label": "Text:",
            "voice_sample_frame": "Voice sample",
            "ref_file_label": "File:",
            "ref_record_label": "Recording:",
            "ref_record_button_start": "Record",
            "ref_record_button_stop": "Stop",
            "output_audio_label": "Output file path:",
            "synthesize_button": "Synthesize",
            "tts_status_ready": "Ready",
            "tts_status_loading_model": "Loading TTS model...",
            "tts_status_translating": "Translating text...",
            "tts_status_synthesizing": "Synthesizing...",
            "tts_status_error": "Error",
            "player_frame": "Playback example",
            "player_no_file": "No file selected",
            "play_button": "Play",
            "stop_button": "Stop",
            "translation_lang_frame": "Translation settings",
            "source_lang_label": "Source language:",
            "target_lang_label": "Target language:",
            "translate_checkbox": "Translate text",
            "synthesis_lang_label": "Synthesis language:",
            "menu_language": "Language",
            "menu_download": "Download",
            "menu_download_whisper": "Download all Whisper models",
            "menu_download_argos": "Download all Argos language packages",
            "downloading_whisper": "Downloading Whisper model: {}",
            "downloading_whisper_complete": "All Whisper models have been downloaded.",
            "downloading_argos": "Downloading all language packages...",
            "downloading_argos_progress": "Downloading {}→{} ({}/{})",
            "downloading_argos_complete": "All available language packages have been downloaded.",
            "no_packages_available": "No packages available.",
            "warning_title": "Warning",
            "error_title": "Critical error",
            "info_title": "Information",
            "question_title": "Confirmation",
            "warning_no_audio_file": "Audio file not found.",
            "warning_no_text_file": "Text file not found.",
            "warning_no_reference": "Voice sample not found.",
            "warning_enter_text": "Enter text to synthesize.",
            "warning_no_file_to_play": "No file to play.",
            "error_read_file": "Failed to read file: {}",
            "error_overwrite_file": "Failed to overwrite {}. Possibly open in another program.",
            "error_play_audio": "Failed to play: {}",
            "error_tts_load": "TTS load error:\n{}",
            "error_package_not_found": "Translation package for {0} -> {1} not found.\nTranslation will be disabled.",
            "question_install_package": "To translate from {} to {} you need to download the language package.\nThis will require internet (once).\nInstall now?",
            "package_install_success": "Translation package {} -> {} successfully installed.",
            "package_install_fail": "Failed to install package: {}",
            "stt_fallback_translation_unavailable": "(Translation unavailable: package not installed)",
            "default_output_txt": "output_{}.txt",
            "default_output_wav": "output_{}.wav",
            "default_reference_voice": "my_voice.wav",
            "recorded_input_prefix": "recorded_input_",
            "recorded_reference_prefix": "recorded_reference_",
            "record_status_recording": "Recording...",
            "record_status_no_data": "No data",
            "chain_package_question": "Direct package {0}->{1} not found.\nDo you want to install the chain via English?\nThis may affect translation quality.",
            "package_not_found_repo": "Translation package {0}->{1} not found in repository.",
            "install_chain_from": "Installing {0}->{1} (via English)",
            "downloading_package": "Downloading package...",
            "installing_package": "Installing package...",
            "ffmpeg_help": "\n" + "="*60 + "\nFFMPEG SOLUTION:\n" + "="*60 + "\n1. Install FFmpeg (full-shared) into 'ffmpeg' folder next to the program.\n2. Ensure ffmpeg.exe and all DLLs are present.\n" + "="*60 + "\n",
            "select_file_title": "Select audio file",
            "save_file_title": "Save file",
            "legal_warning_title": "Legal notice",
            "legal_warning_message": "This software is intended for legitimate use only (e.g. voice dubbing for accessibility, creative projects).\nVoice cloning without the consent of the voice owner or for fraudulent purposes is illegal.\nBy using this software you assume full legal responsibility for any misuse.\n\nA digital watermark is embedded into every generated audio file for forensic identification."
        }
        with open(os.path.join(locales_dir, "en.json"), 'w', encoding='utf-8') as f:
            json.dump(default_en, f, ensure_ascii=False, indent=2)

loc = Localization()
TEXTS = loc.load_language()

# ========== НАСТРОЙКИ / SETTINGS ==========
def load_settings():
    default_settings = {
        "interface_language": loc._current_lang,
        "whisper_model": "small",
        "stt_source_lang": "Russian",
        "stt_target_lang": "English",
        "tts_source_lang": "Russian",
        "tts_target_lang": "English",
        "translate_enabled": True,
        "rec_mode": "file",
        "synth_mode": "text",
        "ref_mode": "file",
        "warning_acknowledged": False
    }
    if not os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(default_settings, f, ensure_ascii=False, indent=2)
        return default_settings
    try:
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            saved = json.load(f)
        for k, v in default_settings.items():
            if k not in saved:
                saved[k] = v
        return saved
    except:
        return default_settings

def save_settings(settings):
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    except:
        pass

# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ / HELPER FUNCTIONS ==========
def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(TEXTS.get("ffmpeg_help", "FFmpeg not found"))
        raise RuntimeError("FFmpeg not found")

# ========== ПЕРЕВОДЧИК / TRANSLATOR ==========
class Translator:
    SUPPORTED_LANGUAGES = {
        "Russian":    {"whisper": "ru", "tts": "ru", "argos": "ru"},
        "English":    {"whisper": "en", "tts": "en", "argos": "en"},
        "Spanish":    {"whisper": "es", "tts": "es", "argos": "es"},
        "French":     {"whisper": "fr", "tts": "fr", "argos": "fr"},
        "German":     {"whisper": "de", "tts": "de", "argos": "de"},
        "Italian":    {"whisper": "it", "tts": "it", "argos": "it"},
        "Portuguese": {"whisper": "pt", "tts": "pt", "argos": "pt"},
        "Polish":     {"whisper": "pl", "tts": "pl", "argos": "pl"},
        "Turkish":    {"whisper": "tr", "tts": "tr", "argos": "tr"},
        "Chinese":    {"whisper": "zh", "tts": "zh-cn", "argos": "zh"},
        "Japanese":   {"whisper": "ja", "tts": "ja", "argos": "ja"},
        "Korean":     {"whisper": "ko", "tts": "ko", "argos": "ko"},
        "Dutch":      {"whisper": "nl", "tts": "nl", "argos": "nl"},
        "Czech":      {"whisper": "cs", "tts": "cs", "argos": "cs"},
        "Arabic":     {"whisper": "ar", "tts": "ar", "argos": "ar"},
        "Hungarian":  {"whisper": "hu", "tts": "hu", "argos": "hu"},
        "Hindi":      {"whisper": "hi", "tts": "hi", "argos": "hi"},
    }

    def __init__(self):
        self.installed_packages = {}
        self._package_availability_cache = {}

    def get_language_list(self):
        return list(self.SUPPORTED_LANGUAGES.keys())

    def get_language_code(self, lang_name, service):
        if lang_name not in self.SUPPORTED_LANGUAGES:
            return None
        return self.SUPPORTED_LANGUAGES[lang_name].get(service)

    def _refresh_installed_languages(self):
        """Принудительно перезагружает список установленных языков.
        Forcefully reloads the list of installed languages."""
        try:
            # Очищаем внутренний кэш Argos Translate
            # Clear the internal cache of Argos Translate
            if hasattr(argostranslate.translate, '_load_installed_languages'):
                argostranslate.translate._load_installed_languages.cache_clear()
            # Сбрасываем наш кэш
            # Reset our cache
            self.installed_packages.clear()
            self._package_availability_cache.clear()
            # Вызываем принудительное обновление
            # Force an update
            argostranslate.translate.get_installed_languages()
        except Exception as e:
            print(f"Warning: failed to refresh languages: {e}")

    def is_package_installed(self, from_lang, to_lang):
        from_code = self.get_language_code(from_lang, "argos")
        to_code = self.get_language_code(to_lang, "argos")
        if not from_code or not to_code:
            return False
        key = (from_code, to_code)
        if key in self.installed_packages:
            return self.installed_packages[key]
        available = argostranslate.translate.get_installed_languages()
        from_lang_obj = next((lang for lang in available if lang.code == from_code), None)
        to_lang_obj = next((lang for lang in available if lang.code == to_code), None)
        if from_lang_obj and to_lang_obj and from_lang_obj.get_translation(to_lang_obj):
            self.installed_packages[key] = True
            return True
        self.installed_packages[key] = False
        return False

    def package_exists(self, from_lang, to_lang):
        from_code = self.get_language_code(from_lang, "argos")
        to_code = self.get_language_code(to_lang, "argos")
        if not from_code or not to_code:
            return False
        key = (from_code, to_code)
        if key in self._package_availability_cache:
            return self._package_availability_cache[key]
        try:
            available = argostranslate.package.get_available_packages()
            exists = any(pkg.from_code == from_code and pkg.to_code == to_code for pkg in available)
            self._package_availability_cache[key] = exists
            return exists
        except:
            return False

    def install_package(self, from_lang, to_lang, progress_callback=None):
        from_code = self.get_language_code(from_lang, "argos")
        to_code = self.get_language_code(to_lang, "argos")
        if not from_code or not to_code:
            raise ValueError(f"Unsupported language pair {from_lang}->{to_lang}")
        available_packages = argostranslate.package.get_available_packages()
        package = next(
            (pkg for pkg in available_packages if pkg.from_code == from_code and pkg.to_code == to_code),
            None
        )
        if not package:
            raise Exception(f"Package {from_lang}->{to_lang} not found in repository")
        if progress_callback:
            progress_callback(0, TEXTS.get("downloading_package", "Downloading package..."))
        download_path = package.download()
        if progress_callback:
            progress_callback(50, TEXTS.get("installing_package", "Installing package..."))
        argostranslate.package.install_from_path(download_path)
        self.installed_packages[(from_code, to_code)] = True
        self._refresh_installed_languages()
        if progress_callback:
            progress_callback(100, TEXTS.get("package_install_success", "Package installed"))
        return True

    def is_translation_available(self, from_lang, to_lang):
        """Проверяет, доступен ли перевод (прямой или через английский).
        Checks whether translation is available (direct or through English)"""
        if self.is_package_installed(from_lang, to_lang):
            return True
        # Проверяем цепочку через английский / Checking chain through English
        if self.is_package_installed(from_lang, "English") and self.is_package_installed("English", to_lang):
            return True
        return False

    def translate(self, text, from_lang, to_lang):
        """Переводит текст, при необходимости используя цепочку через английский.
        Translates text, using English as a bridge if direct package is missing."""
        if not text or not text.strip():
            return text
        if from_lang == to_lang:
            return text
        from_code = self.get_language_code(from_lang, "argos")
        to_code = self.get_language_code(to_lang, "argos")
        if not from_code or not to_code:
            raise ValueError(f"Unknown language codes: {from_lang} -> {to_lang}")

        # Прямой перевод / Direct translation
        if self.is_package_installed(from_lang, to_lang):
            return argostranslate.translate.translate(text, from_code, to_code)

        # Цепочка через английский / Chain via English
        en_code = "en"
        if self.is_package_installed(from_lang, "English") and self.is_package_installed("English", to_lang):
            intermediate = argostranslate.translate.translate(text, from_code, en_code)
            return argostranslate.translate.translate(intermediate, en_code, to_code)

        # Прямой пакет отсутствует, цепочка не настроена → сообщаем об ошибке
        # No direct package, chain not available → raise error
        raise RuntimeError(f"No translation path found from {from_lang} to {to_lang}. "
                           f"Please install the direct package or ensure both {from_lang}->English and English->{to_lang} packages are installed.")

# ========== РАСПОЗНАВАНИЕ РЕЧИ / SPEECH RECOGNITION ==========
class SpeechRecognizer:
    def __init__(self, model_size="small"):
        self.model_size = model_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.vad = webrtcvad.Vad(2)
        self.target_sr = 16000

    def load_model(self, progress_callback=None):
        if self.model is not None:
            return
        whisper_cache = os.path.join(CACHE_DIR, "whisper_cache")
        os.makedirs(whisper_cache, exist_ok=True)
        print(f"Loading Whisper model ({self.model_size})...")
        if progress_callback:
            progress_callback(10, f"Loading {self.model_size} model...")
        self.model = whisper.load_model(
            self.model_size,
            device=self.device,
            download_root=whisper_cache
        )
        if progress_callback:
            progress_callback(100, "Model loaded")
        print("Model loaded.")

    def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Whisper model unloaded from GPU.")

    def _load_audio(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
        audio = waveform.numpy().flatten().astype(np.float32)
        return audio, self.target_sr

    def _normalize_loudness(self, audio, target_dbfs=-20):
        rms = np.sqrt(np.mean(audio**2))
        if rms < 1e-6:
            return audio
        target_rms = 10 ** (target_dbfs / 20)
        gain = target_rms / rms
        audio = audio * gain
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val * 0.95
        return audio

    def _vad_trim(self, audio, sr, frame_duration_ms=30, padding_ms=300):
        audio_int16 = (audio * 32767).astype(np.int16)
        frame_size = int(sr * frame_duration_ms / 1000)
        frames = []
        for i in range(0, len(audio_int16), frame_size):
            frame = audio_int16[i:i+frame_size]
            if len(frame) == frame_size:
                frames.append(frame)
        speech_flags = []
        for frame in frames:
            is_speech = self.vad.is_speech(frame.tobytes(), sr)
            speech_flags.append(is_speech)
        speech_indices = [i for i, flag in enumerate(speech_flags) if flag]
        if not speech_indices:
            return audio
        start_frame = max(0, speech_indices[0] - padding_ms // frame_duration_ms)
        end_frame = min(len(frames), speech_indices[-1] + padding_ms // frame_duration_ms + 1)
        start_sample = start_frame * frame_size
        end_sample = min(len(audio_int16), end_frame * frame_size)
        trimmed_int16 = audio_int16[start_sample:end_sample]
        trimmed_audio = trimmed_int16.astype(np.float32) / 32767.0
        return trimmed_audio

    def recognize(self, audio_path, language_code="ru"):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        audio, sr = self._load_audio(audio_path)
        audio = self._vad_trim(audio, sr)
        if len(audio) < 0.5 * sr:
            print("VAD no speech, using original recording.")
            audio, sr = self._load_audio(audio_path)
        audio = self._normalize_loudness(audio)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        sf.write(temp_path, audio, sr, subtype='PCM_16')
        try:
            result = self.model.transcribe(temp_path, language=language_code, fp16=torch.cuda.is_available())
            text = result["text"].strip()
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        return text

# ========== СИНТЕЗ РЕЧИ / TTS SYNTHESIS ==========
class VoiceCloningSystem:
    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2", progress_callback=None):
        check_ffmpeg()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.default_reference = os.path.join(REF_SAMPLES_DIR, TEXTS.get("default_reference_voice", "my_voice.wav"))
        print("Loading TTS model...")
        if progress_callback:
            progress_callback(10, TEXTS.get("tts_status_loading_model", "Loading TTS model..."))
        try:
            self.tts = TTS(model_name=model_name, progress_bar=True).to(self.device)
            if progress_callback:
                progress_callback(100, "Model loaded")
            print("Model loaded successfully!")
        except Exception as e:
            if progress_callback:
                progress_callback(100, f"Error: {e}")
            print(f"Model load error: {e}")
            raise

    def record_voice_sample(self, filename: str = None, duration: int = 10, sr: int = 16000) -> str:
        if filename is None:
            filename = self.default_reference
        print(f"Recording {duration} seconds...")
        try:
            recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
            for i in range(duration):
                time.sleep(1)
                print(f"Recorded: {i+1}/{duration} sec.")
            sd.wait()
            recording = recording.flatten()
            recording_int16 = np.int16(recording * 32767)
            with wave.open(filename, 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sr)
                wav_file.writeframes(recording_int16.tobytes())
            print(f"Recording saved to: {filename}")
            return self._simple_preprocess_wav(filename)
        except Exception as e:
            print(f"Recording error: {e}")
            return None

    def _simple_preprocess_wav(self, input_path: str) -> str:
        try:
            with wave.open(input_path, 'r') as wav_file:
                params = wav_file.getparams()
                frames = wav_file.readframes(params.nframes)
                audio_data = np.frombuffer(frames, dtype=np.int16)
            audio_float = audio_data.astype(np.float32) / 32768.0
            threshold = 0.01
            abs_audio = np.abs(audio_float)
            start = 0
            for i in range(0, len(audio_float), 160):
                if np.max(abs_audio[i:i+160]) > threshold:
                    start = max(0, i - 320)
                    break
            end = len(audio_float)
            for i in range(len(audio_float)-1, 0, -160):
                if np.max(abs_audio[i-160:i]) > threshold:
                    end = min(len(audio_float), i + 320)
                    break
            if end - start > 16000:
                audio_float = audio_float[start:end]
            max_samples = 15 * 16000
            if len(audio_float) > max_samples:
                audio_float = audio_float[:max_samples]
            max_val = np.max(np.abs(audio_float))
            if max_val > 0:
                audio_float = audio_float / max_val
            audio_int16 = np.int16(audio_float * 32767)
            base_name = os.path.splitext(input_path)[0]
            output_path = f"{base_name}_processed.wav"
            with wave.open(output_path, 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(audio_int16.tobytes())
            return output_path
        except Exception as e:
            print(f"Audio preprocessing failed: {e}")
            return input_path

    def _embed_watermark(self, file_path: str):
        """Встраивает неслышимый цифровой водяной знак (LSB) в WAV файл.
        Embeds an inaudible digital watermark (LSB) into the WAV file."""
        try:
            # Открываем файл для чтения
            # Open the file for reading
            with wave.open(file_path, 'rb') as wav:
                params = wav.getparams()
                frames = wav.readframes(params.nframes)
            # Преобразуем в numpy массив int16
            # Convert to numpy int16 array
            audio = np.frombuffer(frames, dtype=np.int16).copy()
            if len(audio) == 0:
                return
            # Сигнатура водяного знака "VCLONE" (6 байт = 48 бит)
            # Watermark signature "VCLONE" (6 bytes = 48 bits)
            signature = "VCLONE"
            bits = []
            for ch in signature:
                # младший бит первым (LSB first)
                # least significant bit first (LSB first)
                for i in range(8):
                    bits.append((ord(ch) >> i) & 1)
            # Встраиваем в первые min(48, len(audio)) отсчетов
            # Embed into the first min(48, len(audio)) samples
            n_bits = min(len(bits), len(audio))
            for i in range(n_bits):
                audio[i] = (audio[i] & 0xFE) | bits[i]   # обнуляем младший бит и ставим нужный / zero the LSB and set the needed bit
            # Записываем обратно
            # Write back
            with wave.open(file_path, 'wb') as wav:
                wav.setparams(params)
                wav.writeframes(audio.tobytes())
            print("Watermark embedded successfully.")
        except Exception as e:
            print(f"Failed to embed watermark: {e}")

    def clone_voice(self, text: str, reference_audio_path: str, output_path: str = "output.wav", language: str = "ru") -> str:
        if not os.path.exists(reference_audio_path):
            raise FileNotFoundError(f"File {reference_audio_path} not found")
        print("Starting synthesis...")
        try:
            self.tts.tts_to_file(
                text=text,
                speaker_wav=reference_audio_path,
                language=language,
                file_path=output_path
            )
        except Exception as e:
            print(f"Synthesis error: {e}")
            raise
        if os.path.exists(output_path):
            # Внедряем водяной знак
            # Embed watermark
            self._embed_watermark(output_path)
            return output_path
        else:
            raise RuntimeError("Output file not created")

# ========== ЗАПИСЬ / RECORDER ==========
class Recorder:
    def __init__(self, entry_widget, var, status_entry, sample_rate=16000, channels=1):
        self.entry = entry_widget
        self.var = var
        self.status_entry = status_entry
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording = False
        self.audio_data = []
        self.stream = None
        self.thread = None

    def start_recording(self):
        self.recording = True
        self.audio_data = []
        self.status_entry.config(state='normal')
        self.status_entry.delete(0, tk.END)
        self.status_entry.insert(0, TEXTS.get("record_status_recording", "Recording..."))
        self.status_entry.config(state='readonly')
        self.thread = threading.Thread(target=self._record)
        self.thread.start()

    def _record(self):
        try:
            def callback(indata, frames, time, status):
                if self.recording:
                    self.audio_data.append(indata.copy())
            self.stream = sd.InputStream(samplerate=self.sample_rate,
                                         channels=self.channels,
                                         callback=callback)
            self.stream.start()
            while self.recording:
                sd.sleep(100)
        except Exception as e:
            self.status_entry.config(state='normal')
            self.status_entry.delete(0, tk.END)
            self.status_entry.insert(0, f"Error: {e}")
            self.status_entry.config(state='readonly')
        finally:
            if self.stream:
                self.stream.stop()
                self.stream.close()

    def stop_recording(self, filename):
        self.recording = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
        if self.audio_data:
            audio = np.concatenate(self.audio_data, axis=0)
            audio_int16 = (audio * 32767).astype(np.int16)
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_int16.tobytes())
            self.status_entry.config(state='normal')
            self.status_entry.delete(0, tk.END)
            self.status_entry.insert(0, filename)
            self.status_entry.config(state='readonly')
            self.var.set(filename)
            self.entry.delete(0, tk.END)
            self.entry.insert(0, filename)
        else:
            self.status_entry.config(state='normal')
            self.status_entry.delete(0, tk.END)
            self.status_entry.insert(0, TEXTS.get("record_status_no_data", "No data"))
            self.status_entry.config(state='readonly')

# ========== ПАНЕЛЬ ПРОГРЕССА / PROGRESS PANEL ==========
class ProgressOverlay:
    def __init__(self, parent_grid, row):
        self.parent = parent_grid
        self.grid_row = row
        self.frame = None
        self.progress_var = None
        self.progress_bar = None
        self.label = None
        self.is_determinate = False

    def show(self, message="Loading...", mode='indeterminate'):
        self.hide()
        self.frame = ttk.Frame(self.parent, relief='sunken', borderwidth=1)
        self.frame.grid(row=self.grid_row, column=0, sticky="ew", pady=(0,0))
        self.frame.grid_propagate(False)
        self.frame.config(height=100)
        self.label = ttk.Label(self.frame, text=message, anchor='center')
        self.label.pack(pady=(10,5))
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.frame, variable=self.progress_var, mode=mode)
        self.progress_bar.pack(fill=tk.X, padx=20, pady=5)
        self.is_determinate = (mode == 'determinate')
        if mode == 'indeterminate':
            self.progress_bar.start(10)
        self.parent.update_idletasks()

    def update(self, value, text=None):
        if self.frame is None:
            return
        if self.is_determinate:
            self.progress_var.set(value)
        if text:
            self.label.config(text=text)
        self.parent.update_idletasks()

    def hide(self):
        if self.frame:
            if self.progress_bar and not self.is_determinate:
                self.progress_bar.stop()
            self.frame.destroy()
            self.frame = None
            self.progress_bar = None
            self.progress_var = None
            self.label = None

# ========== ОСНОВНОЙ GUI / MAIN GUI ==========
class App:
    def __init__(self, root):
        global TEXTS
        self.root = root
        self.settings = load_settings()
        # Показываем предупреждение о законности использования (один раз)
        # Show legal warning once
        if not self.settings.get("warning_acknowledged", False):
            messagebox.showwarning(
                TEXTS.get("legal_warning_title", "Legal notice"),
                TEXTS.get("legal_warning_message", "This software is intended for legitimate use only...")
            )
            self.settings["warning_acknowledged"] = True
            save_settings(self.settings)

        self.translator = Translator()
        self.stt_engine = SpeechRecognizer(model_size=self.settings.get("whisper_model", "small"))
        self.tts_engine = None
        self._loading_tts = False

        # Переменные / Variables
        self.stt_input_path = tk.StringVar()
        self.stt_output_path = tk.StringVar()
        self.tts_text_file = tk.StringVar()
        self.tts_reference_path = tk.StringVar()
        self.tts_output_path = tk.StringVar()
        self.last_synthesized = tk.StringVar()

        self.stt_status = tk.StringVar(value=TEXTS.get("stt_status_ready", "Ready"))
        self.tts_status = tk.StringVar(value=TEXTS.get("tts_status_ready", "Ready"))

        self.rec_mode = tk.StringVar(value=self.settings.get("rec_mode", "file"))
        self.synth_mode = tk.StringVar(value=self.settings.get("synth_mode", "text"))
        self.ref_mode = tk.StringVar(value=self.settings.get("ref_mode", "file"))
        self.stt_model_size = tk.StringVar(value=self.settings.get("whisper_model", "small"))
        self.translate_enabled = tk.BooleanVar(value=self.settings.get("translate_enabled", True))
        self.stt_source_lang = tk.StringVar(value=self.settings.get("stt_source_lang", "Russian"))
        self.stt_target_lang = tk.StringVar(value=self.settings.get("stt_target_lang", "English"))
        self.tts_source_lang = tk.StringVar(value=self.settings.get("tts_source_lang", "Russian"))
        self.tts_target_lang = tk.StringVar(value=self.settings.get("tts_target_lang", "English"))
        self.interface_lang = tk.StringVar(value=self.settings.get("interface_language", loc._current_lang))

        self.is_recording_stt = False
        self.is_recording_ref = False
        self.recorder = None
        self.ref_recorder = None

        # Настройка сетки корневого окна / Root grid configuration
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)
        self.root.grid_columnconfigure(0, weight=1)

        self.main_container = ttk.Frame(self.root)
        self.main_container.grid(row=0, column=0, sticky="nsew")
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)

        self.progress_overlay = ProgressOverlay(self.root, row=1)

        self.interface_lang.set(self.settings.get("interface_language", "en"))
        TEXTS = loc.load_language(self.interface_lang.get())

        self.create_widgets()
        self.load_tts_model_background()
        self.bind_events()
        self.update_rec_mode()
        self.update_synth_mode()
        self.update_ref_mode()
        self.update_buttons_state()

    def bind_events(self):
        self.stt_input_path.trace_add('write', lambda *a: self.update_buttons_state())
        self.tts_reference_path.trace_add('write', lambda *a: self.update_buttons_state())
        self.rec_mode.trace_add('write', lambda *a: self.update_rec_mode())
        self.synth_mode.trace_add('write', lambda *a: self.update_synth_mode())
        self.ref_mode.trace_add('write', lambda *a: self.update_ref_mode())
        self.stt_model_size.trace_add('write', lambda *a: self.on_model_size_change())
        self.translate_enabled.trace_add('write', lambda *a: self.update_buttons_state())
        self.stt_source_lang.trace_add('write', lambda *a: self.update_buttons_state())
        self.stt_target_lang.trace_add('write', lambda *a: self.update_buttons_state())
        self.tts_source_lang.trace_add('write', lambda *a: self.update_buttons_state())
        self.tts_target_lang.trace_add('write', lambda *a: self.update_buttons_state())

    def on_interface_language_change(self):
        global TEXTS
        new_lang = self.interface_lang.get()
        TEXTS = loc.load_language(new_lang)
        self.settings["interface_language"] = new_lang
        save_settings(self.settings)
        self.stt_status.set(TEXTS.get("stt_status_ready", "Ready"))
        self.tts_status.set(TEXTS.get("tts_status_ready", "Ready"))
        self.rebuild_ui(preserve_tts=True)

    def rebuild_ui(self, preserve_tts=False):
        tts_saved = self.tts_engine if preserve_tts else None
        for widget in self.main_container.winfo_children():
            widget.destroy()
        self.create_widgets()
        self.bind_events()
        if tts_saved is not None:
            self.tts_engine = tts_saved
            self.tts_loaded()
        else:
            self.load_tts_model_background()
        self.update_rec_mode()
        self.update_synth_mode()
        self.update_ref_mode()
        self.update_buttons_state()

    def create_widgets(self):
        self.root.title(TEXTS.get("app_title", "VoiceClone-GUI"))
        self.root.geometry("1280x800")
        self.root.minsize(1280, 800)

        menubar = tk.Menu(self.root)

        lang_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label=TEXTS.get("menu_language", "Language"), menu=lang_menu)
        available_langs = loc.get_available_languages()
        for lang in available_langs:
            lang_menu.add_radiobutton(label=lang.upper(), variable=self.interface_lang, value=lang,
                                    command=self.on_interface_language_change)

        download_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label=TEXTS.get("menu_download", "Download"), menu=download_menu)
        download_menu.add_command(label=TEXTS.get("menu_download_whisper", "Download all Whisper models"),
                                  command=self.download_all_whisper_models)
        download_menu.add_command(label=TEXTS.get("menu_download_argos", "Download all Argos language packages"),
                                  command=self.download_all_argos_packages)

        self.root.config(menu=menubar)

        paned = ttk.Frame(self.main_container)
        paned.grid(row=0, column=0, sticky="nsew")
        paned.grid_columnconfigure(0, weight=1)
        paned.grid_columnconfigure(1, weight=1)
        paned.grid_rowconfigure(0, weight=1)

        # ================= ЛЕВАЯ КОЛОНКА / LEFT COLUMN =================
        left_frame = ttk.LabelFrame(paned, text=TEXTS.get("left_panel_title", "Speech Recognition"), padding="10")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        left_frame.grid_columnconfigure(1, weight=1)
        left_frame.grid_rowconfigure(7, weight=1)
        left_frame.grid_rowconfigure(8, weight=1)

        mode_frame = ttk.Frame(left_frame)
        mode_frame.grid(row=0, column=0, columnspan=4, sticky="w", pady=2)
        ttk.Radiobutton(mode_frame, text=TEXTS.get("mode_file", "Select file"), variable=self.rec_mode,
                        value="file").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text=TEXTS.get("mode_record", "Record file"), variable=self.rec_mode,
                        value="record").pack(side=tk.LEFT, padx=5)

        ttk.Label(left_frame, text=TEXTS.get("file_label", "File:")).grid(row=1, column=0, sticky="w", pady=2)
        self.entry_stt_input = ttk.Entry(left_frame, textvariable=self.stt_input_path, width=35)
        self.entry_stt_input.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        self.stt_browse_btn = ttk.Button(left_frame, text=TEXTS.get("browse_button", "Browse"),
                                        command=lambda: self.select_file(self.stt_input_path, [("WAV files", "*.wav")]))
        self.stt_browse_btn.grid(row=1, column=2, padx=2)

        ttk.Label(left_frame, text=TEXTS.get("record_label", "Recording:")).grid(row=2, column=0, sticky="w", pady=2)
        self.record_status_entry = ttk.Entry(left_frame, width=35, state='readonly')
        self.record_status_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=2)
        self.record_btn = ttk.Button(left_frame, text=TEXTS.get("record_button_start", "Record"), command=self.toggle_record)
        self.record_btn.grid(row=2, column=2, padx=2)

        ttk.Label(left_frame, text=TEXTS.get("output_file_label", "Output file path:")).grid(row=3, column=0, sticky="w", pady=2)
        entry_stt_out = ttk.Entry(left_frame, textvariable=self.stt_output_path, width=35)
        entry_stt_out.grid(row=3, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(left_frame, text=TEXTS.get("browse_button", "Browse"),
                command=lambda: self.select_save_file(self.stt_output_path, [("Text files", "*.txt")], ".txt")).grid(row=3, column=2, padx=2)

        model_frame = ttk.LabelFrame(left_frame, text=TEXTS.get("whisper_model_frame", "Whisper model"), padding="5")
        model_frame.grid(row=4, column=0, columnspan=4, sticky="ew", pady=5)
        sizes = [("tiny", "tiny"), ("base", "base"), ("small", "small"), ("medium", "medium"), ("large", "large")]
        for i, (label, val) in enumerate(sizes):
            rb = ttk.Radiobutton(model_frame, text=label, variable=self.stt_model_size, value=val)
            rb.grid(row=0, column=i, padx=5, sticky="w")

        stt_trans_frame = ttk.LabelFrame(left_frame, text=TEXTS.get("translation_lang_frame", "Translation settings"), padding="5")
        stt_trans_frame.grid(row=5, column=0, columnspan=4, sticky="ew", pady=5)
        stt_trans_frame.grid_columnconfigure(0, weight=0)
        stt_trans_frame.grid_columnconfigure(1, weight=1)
        stt_trans_frame.grid_columnconfigure(2, weight=0)
        stt_trans_frame.grid_columnconfigure(3, weight=1)

        ttk.Label(stt_trans_frame, text=TEXTS.get("source_lang_label", "Source language:")).grid(row=0, column=0, sticky="w", padx=5)
        lang_list = self.translator.get_language_list()
        self.stt_source_combo = ttk.Combobox(stt_trans_frame, textvariable=self.stt_source_lang, values=lang_list, state="readonly", width=12)
        self.stt_source_combo.grid(row=0, column=1, sticky="ew", padx=5)

        ttk.Label(stt_trans_frame, text=TEXTS.get("target_lang_label", "Target language:")).grid(row=0, column=2, sticky="w", padx=5)
        self.stt_target_combo = ttk.Combobox(stt_trans_frame, textvariable=self.stt_target_lang, values=lang_list, state="readonly", width=12)
        self.stt_target_combo.grid(row=0, column=3, sticky="ew", padx=5)

        self.translate_check = ttk.Checkbutton(stt_trans_frame, text=TEXTS.get("translate_checkbox", "Translate text"),
                                            variable=self.translate_enabled)
        self.translate_check.grid(row=1, column=0, columnspan=4, sticky="w", padx=5, pady=2)

        self.recognize_btn = ttk.Button(left_frame, text=TEXTS.get("recognize_translate_button", "Recognize & Translate"),
                                        command=self.recognize, state=tk.DISABLED)
        self.recognize_btn.grid(row=6, column=1, pady=10, sticky="w")
        ttk.Label(left_frame, textvariable=self.stt_status).grid(row=6, column=2, sticky="w")

        text_frame = ttk.LabelFrame(left_frame, text=TEXTS.get("original_text_frame", "Recognized text"), padding="5")
        text_frame.grid(row=7, column=0, columnspan=4, sticky="nsew", pady=5)
        text_container = ttk.Frame(text_frame)
        text_container.pack(fill=tk.BOTH, expand=True)
        self.recognized_text = tk.Text(text_container, wrap=tk.WORD, state=tk.DISABLED)
        scroll_orig = ttk.Scrollbar(text_container, orient=tk.VERTICAL, command=self.recognized_text.yview)
        self.recognized_text.configure(yscrollcommand=scroll_orig.set)
        scroll_orig.pack(side=tk.RIGHT, fill=tk.Y)
        self.recognized_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        translation_frame = ttk.LabelFrame(left_frame, text=TEXTS.get("translated_text_frame", "Translation"), padding="5")
        translation_frame.grid(row=8, column=0, columnspan=4, sticky="nsew", pady=5)
        trans_container = ttk.Frame(translation_frame)
        trans_container.pack(fill=tk.BOTH, expand=True)
        self.translated_text = tk.Text(trans_container, wrap=tk.WORD, state=tk.DISABLED)
        scroll_trans = ttk.Scrollbar(trans_container, orient=tk.VERTICAL, command=self.translated_text.yview)
        self.translated_text.configure(yscrollcommand=scroll_trans.set)
        scroll_trans.pack(side=tk.RIGHT, fill=tk.Y)
        self.translated_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ================= ПРАВАЯ КОЛОНКА / RIGHT COLUMN =================
        right_frame = ttk.Frame(paned)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        right_frame.grid_columnconfigure(0, weight=1)

        tts_frame = ttk.LabelFrame(right_frame, text=TEXTS.get("right_panel_title", "Speech Synthesis"), padding="10")
        tts_frame.pack(fill=tk.X, pady=5, expand=False)
        tts_frame.grid_columnconfigure(1, weight=1)

        mode_synth_frame = ttk.Frame(tts_frame)
        mode_synth_frame.grid(row=0, column=0, columnspan=3, sticky="w", pady=2)
        ttk.Radiobutton(mode_synth_frame, text=TEXTS.get("synth_mode_file", "Select file"), variable=self.synth_mode,
                        value="file").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_synth_frame, text=TEXTS.get("synth_mode_text", "Enter text"), variable=self.synth_mode,
                        value="text").pack(side=tk.LEFT, padx=5)

        ttk.Label(tts_frame, text=TEXTS.get("text_file_label", "Text file:")).grid(row=1, column=0, sticky="w", pady=2)
        self.entry_tts_file = ttk.Entry(tts_frame, textvariable=self.tts_text_file, width=35)
        self.entry_tts_file.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        self.tts_browse_btn = ttk.Button(tts_frame, text=TEXTS.get("browse_button", "Browse"),
                                        command=lambda: self.select_file(self.tts_text_file, [("Text files", "*.txt")]))
        self.tts_browse_btn.grid(row=1, column=2, padx=2)

        ttk.Label(tts_frame, text=TEXTS.get("text_label", "Text:")).grid(row=2, column=0, sticky="nw", pady=2)
        text_container_synth = ttk.Frame(tts_frame)
        text_container_synth.grid(row=2, column=1, columnspan=2, sticky="ew", pady=2)
        self.tts_text_entry = tk.Text(text_container_synth, height=8, wrap=tk.WORD)
        scroll_synth = ttk.Scrollbar(text_container_synth, orient=tk.VERTICAL, command=self.tts_text_entry.yview)
        self.tts_text_entry.configure(yscrollcommand=scroll_synth.set)
        scroll_synth.pack(side=tk.RIGHT, fill=tk.Y)
        self.tts_text_entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tts_lang_frame = ttk.Frame(tts_frame)
        tts_lang_frame.grid(row=3, column=0, columnspan=3, sticky="w", pady=5)
        ttk.Label(tts_lang_frame, text=TEXTS.get("source_lang_label", "Source language:")).pack(side=tk.LEFT, padx=5)
        self.tts_source_combo = ttk.Combobox(tts_lang_frame, textvariable=self.tts_source_lang, values=lang_list, state="readonly", width=12)
        self.tts_source_combo.pack(side=tk.LEFT, padx=5)
        ttk.Label(tts_lang_frame, text=TEXTS.get("synthesis_lang_label", "Synthesis language:")).pack(side=tk.LEFT, padx=5)
        self.tts_target_combo = ttk.Combobox(tts_lang_frame, textvariable=self.tts_target_lang, values=lang_list, state="readonly", width=12)
        self.tts_target_combo.pack(side=tk.LEFT, padx=5)

        ref_frame = ttk.LabelFrame(right_frame, text=TEXTS.get("voice_sample_frame", "Voice sample"), padding="10")
        ref_frame.pack(fill=tk.X, pady=5, expand=False)
        ref_frame.grid_columnconfigure(1, weight=1)

        mode_ref_frame = ttk.Frame(ref_frame)
        mode_ref_frame.grid(row=0, column=0, columnspan=3, sticky="w", pady=2)
        ttk.Radiobutton(mode_ref_frame, text=TEXTS.get("mode_file", "Select file"), variable=self.ref_mode,
                        value="file").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_ref_frame, text=TEXTS.get("mode_record", "Record file"), variable=self.ref_mode,
                        value="record").pack(side=tk.LEFT, padx=5)

        ttk.Label(ref_frame, text=TEXTS.get("ref_file_label", "File:")).grid(row=1, column=0, sticky="w", pady=2)
        self.entry_ref = ttk.Entry(ref_frame, textvariable=self.tts_reference_path, width=35)
        self.entry_ref.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        self.ref_browse_btn = ttk.Button(ref_frame, text=TEXTS.get("browse_button", "Browse"),
                                        command=lambda: self.select_file(self.tts_reference_path, [("WAV files", "*.wav")]))
        self.ref_browse_btn.grid(row=1, column=2, padx=2)

        ttk.Label(ref_frame, text=TEXTS.get("ref_record_label", "Recording:")).grid(row=2, column=0, sticky="w", pady=2)
        self.ref_record_status_entry = ttk.Entry(ref_frame, width=35, state='readonly')
        self.ref_record_status_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=2)
        self.ref_record_btn = ttk.Button(ref_frame, text=TEXTS.get("ref_record_button_start", "Record"), command=self.toggle_ref_record)
        self.ref_record_btn.grid(row=2, column=2, padx=2)

        ttk.Label(ref_frame, text=TEXTS.get("output_audio_label", "Output file path:")).grid(row=3, column=0, sticky="w", pady=2)
        entry_tts_out = ttk.Entry(ref_frame, textvariable=self.tts_output_path, width=35)
        entry_tts_out.grid(row=3, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(ref_frame, text=TEXTS.get("browse_button", "Browse"),
                command=lambda: self.select_save_file(self.tts_output_path, [("WAV files", "*.wav")], ".wav")).grid(row=3, column=2, padx=2)

        self.synthesize_btn = ttk.Button(ref_frame, text=TEXTS.get("synthesize_button", "Synthesize"), command=self.synthesize, state=tk.DISABLED)
        self.synthesize_btn.grid(row=4, column=1, pady=10, sticky="w")
        ttk.Label(ref_frame, textvariable=self.tts_status).grid(row=4, column=2, sticky="w")

        player_frame = ttk.LabelFrame(right_frame, text=TEXTS.get("player_frame", "Playback example"), padding="5")
        player_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.player_file_label = ttk.Label(player_frame, text=TEXTS.get("player_no_file", "No file selected"))
        self.player_file_label.pack(pady=5)
        btn_frame = ttk.Frame(player_frame)
        btn_frame.pack()
        self.play_btn = ttk.Button(btn_frame, text=TEXTS.get("play_button", "Play"), command=self.play_audio, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(btn_frame, text=TEXTS.get("stop_button", "Stop"), command=self.stop_audio, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

    # ========== ВЫБОР ФАЙЛА (БЕЗ ЛИШНИХ ДИАЛОГОВ) ==========
    # ========== FILE SELECTION (WITHOUT EXTRA DIALOGS) ==========
    def select_file(self, var, filetypes):
        initial_dir = INPUT_DIR
        if not os.path.exists(initial_dir):
            initial_dir = BASE_DIR
        filename = filedialog.askopenfilename(
            initialdir=initial_dir,
            filetypes=filetypes,
            title=TEXTS.get("select_file_title", "Select file")
        )
        if filename:
            var.set(filename)
            self.update_buttons_state()

    def select_save_file(self, var, filetypes, defaultextension):
        initial_dir = OUTPUT_DIR
        if not os.path.exists(initial_dir):
            initial_dir = BASE_DIR
        timestamp = int(time.time())
        default_filename = TEXTS.get("default_output_txt" if defaultextension==".txt" else "default_output_wav", "output_{}").format(timestamp)
        filename = filedialog.asksaveasfilename(
            initialdir=initial_dir,
            initialfile=default_filename,
            filetypes=filetypes,
            defaultextension=defaultextension,
            title=TEXTS.get("save_file_title", "Save file")
        )
        if filename:
            var.set(filename)
            self.update_buttons_state()

    # ========== ОБНОВЛЕНИЕ СОСТОЯНИЙ / STATE UPDATES ==========
    def update_rec_mode(self):
        if self.is_recording_stt or self.is_recording_ref:
            return
        mode = self.rec_mode.get()
        if mode == "file":
            self.entry_stt_input.config(state=tk.NORMAL)
            self.stt_browse_btn.config(state=tk.NORMAL)
            self.record_btn.config(state=tk.DISABLED)
        else:
            self.entry_stt_input.config(state=tk.DISABLED)
            self.stt_browse_btn.config(state=tk.DISABLED)
            self.record_btn.config(state=tk.NORMAL)

    def update_synth_mode(self):
        if self.is_recording_stt or self.is_recording_ref:
            return
        mode = self.synth_mode.get()
        if mode == "file":
            self.entry_tts_file.config(state=tk.NORMAL)
            self.tts_browse_btn.config(state=tk.NORMAL)
            self.tts_text_entry.config(state=tk.DISABLED)
        else:
            self.entry_tts_file.config(state=tk.DISABLED)
            self.tts_browse_btn.config(state=tk.DISABLED)
            self.tts_text_entry.config(state=tk.NORMAL)

    def update_ref_mode(self):
        if self.is_recording_stt or self.is_recording_ref:
            return
        mode = self.ref_mode.get()
        if mode == "file":
            self.entry_ref.config(state=tk.NORMAL)
            self.ref_browse_btn.config(state=tk.NORMAL)
            self.ref_record_btn.config(state=tk.DISABLED)
        else:
            self.entry_ref.config(state=tk.DISABLED)
            self.ref_browse_btn.config(state=tk.DISABLED)
            self.ref_record_btn.config(state=tk.NORMAL)

    def update_buttons_state(self):
        file_ok = self.stt_input_path.get() and os.path.exists(self.stt_input_path.get())
        if file_ok and not self.is_recording_stt and not self.is_recording_ref:
            self.recognize_btn.config(state=tk.NORMAL)
        else:
            self.recognize_btn.config(state=tk.DISABLED)

        ref_ok = self.tts_reference_path.get() and os.path.exists(self.tts_reference_path.get())
        text_ok = False
        if self.synth_mode.get() == "file":
            text_ok = self.tts_text_file.get() and os.path.exists(self.tts_text_file.get())
        else:
            text_ok = bool(self.tts_text_entry.get(1.0, tk.END).strip())
        if self.tts_engine and ref_ok and text_ok and not self.is_recording_stt and not self.is_recording_ref:
            self.synthesize_btn.config(state=tk.NORMAL)
        else:
            self.synthesize_btn.config(state=tk.DISABLED)

    def set_ui_enabled(self, enable, skip_rec_buttons=False):
        state = tk.NORMAL if enable else tk.DISABLED
        self.stt_browse_btn.config(state=state)
        self.entry_stt_input.config(state=state)
        self.recognize_btn.config(state=state)
        self.tts_browse_btn.config(state=state)
        self.entry_tts_file.config(state=state)
        if enable and self.synth_mode.get() == "text":
            self.tts_text_entry.config(state=tk.NORMAL)
        else:
            self.tts_text_entry.config(state=state)
        self.ref_browse_btn.config(state=state)
        self.entry_ref.config(state=state)
        self.synthesize_btn.config(state=state)
        self.play_btn.config(state=state)
        self.stop_btn.config(state=state)

        if not skip_rec_buttons:
            self.record_btn.config(state=state)
            self.ref_record_btn.config(state=state)
        else:
            self.update_rec_mode()
            self.update_ref_mode()

    # ========== ЗАПИСЬ / RECORDING ==========
    def toggle_record(self):
        if self.is_recording_stt:
            self.recorder.stop_recording(self.current_record_filename)
            self.record_btn.config(text=TEXTS.get("record_button_start", "Record"))
            self.is_recording_stt = False
            self.recorder = None
            self.set_ui_enabled(True, skip_rec_buttons=True)
            self.update_rec_mode()
            self.update_ref_mode()
            self.update_buttons_state()
        else:
            timestamp = int(time.time())
            self.current_record_filename = os.path.join(REF_SAMPLES_DIR, f"recorded_input_{timestamp}.wav")
            self.recorder = Recorder(self.entry_stt_input, self.stt_input_path, self.record_status_entry)
            self.recorder.start_recording()
            self.record_btn.config(text=TEXTS.get("record_button_stop", "Stop"))
            self.is_recording_stt = True
            self.set_ui_enabled(False, skip_rec_buttons=True)
            self.ref_record_btn.config(state=tk.DISABLED)

    def toggle_ref_record(self):
        if self.is_recording_ref:
            self.ref_recorder.stop_recording(self.current_ref_filename)
            self.ref_record_btn.config(text=TEXTS.get("ref_record_button_start", "Record"))
            self.is_recording_ref = False
            self.ref_recorder = None
            self.set_ui_enabled(True, skip_rec_buttons=True)
            self.update_rec_mode()
            self.update_ref_mode()
            self.update_buttons_state()
        else:
            timestamp = int(time.time())
            self.current_ref_filename = os.path.join(REF_SAMPLES_DIR, f"recorded_reference_{timestamp}.wav")
            self.ref_recorder = Recorder(self.entry_ref, self.tts_reference_path, self.ref_record_status_entry)
            self.ref_recorder.start_recording()
            self.ref_record_btn.config(text=TEXTS.get("ref_record_button_stop", "Stop"))
            self.is_recording_ref = True
            self.set_ui_enabled(False, skip_rec_buttons=True)
            self.record_btn.config(state=tk.DISABLED)

    # ========== ЗАГРУЗКА TTS / TTS LOADING ==========
    def load_tts_model_background(self):
        if self._loading_tts:
            return
        self._loading_tts = True
        def update_progress(percent, msg):
            self.progress_overlay.update(percent, msg)
        self.progress_overlay.show(TEXTS.get("tts_status_loading_model", "Loading TTS model..."), mode='determinate')
        threading.Thread(target=self.load_tts_model, args=(update_progress,), daemon=True).start()

    def load_tts_model(self, progress_callback):
        try:
            progress_callback(10, TEXTS.get("tts_status_loading_model", "Loading TTS model..."))
            self.tts_engine = VoiceCloningSystem(progress_callback=progress_callback)
            progress_callback(100, TEXTS.get("tts_loading_done", "Model loaded"))
            self.root.after(0, self.tts_loaded)
        except Exception as e:
            traceback.print_exc()
            self.root.after(0, lambda e=e: self.show_error_and_exit(TEXTS.get("error_tts_load", "TTS load error:\n{}").format(e)))
        finally:
            self._loading_tts = False
            self.root.after(0, self.progress_overlay.hide)

    def tts_loaded(self):
        self.update_buttons_state()

    def on_model_size_change(self):
        new_size = self.stt_model_size.get()
        self.settings["whisper_model"] = new_size
        save_settings(self.settings)
        self.stt_engine = SpeechRecognizer(model_size=new_size)

    def show_error_and_exit(self, message):
        messagebox.showerror(TEXTS.get("error_title", "Critical error"), message)
        self.root.quit()

    # ========== УСТАНОВКА ПАКЕТА ПЕРЕВОДА / INSTALL TRANSLATION PACKAGE ==========
    def _install_translation_package(self, from_lang, to_lang):
        from_code = self.translator.get_language_code(from_lang, "argos")
        to_code = self.translator.get_language_code(to_lang, "argos")
        if not from_code or not to_code:
            return False

        # Если перевод уже доступен (прямой или через английский)
        # If translation is already available (direct or through English)
        if self.translator.is_translation_available(from_lang, to_lang):
            return True

        available = argostranslate.package.get_available_packages()
        direct_package = next((pkg for pkg in available if pkg.from_code == from_code and pkg.to_code == to_code), None)

        if direct_package:
            answer = messagebox.askyesno(
                TEXTS.get("question_title", "Confirmation"),
                TEXTS.get("question_install_package", "To translate from {} to {} you need to download the language package.\nThis will require internet (once).\nInstall now?").format(from_lang, to_lang)
            )
            if not answer:
                return False
            return self._install_single_package(direct_package, from_lang, to_lang)

        # ---------- Цепочка через английский / Chain via English ----------
        en_code = "en"
        en_pkg_from = next((pkg for pkg in available if pkg.from_code == from_code and pkg.to_code == en_code), None)
        en_pkg_to   = next((pkg for pkg in available if pkg.from_code == en_code and pkg.to_code == to_code), None)

        if not en_pkg_from or not en_pkg_to:
            self.root.after(0, lambda: messagebox.showerror(
                TEXTS.get("error_title", "Error"),
                TEXTS.get("package_not_found_repo", "Translation package {0}->{1} not found in repository.").format(from_lang, to_lang)
            ))
            return False

        # Проверяем, какие пакеты уже установлены
        # Check which packages are already installed
        need_from = not self.translator.is_package_installed(from_lang, "English")
        need_to   = not self.translator.is_package_installed("English", to_lang)

        if not need_from and not need_to:
            # Пакеты уже есть, но по какой-то причине is_translation_available вернул False
            # Packages already exist, but for some reason is_translation_available returned False
            # Обновляем кэш и пробуем ещё раз
            # Update cache and try again
            self.translator._refresh_installed_languages()
            return self.translator.is_translation_available(from_lang, to_lang)

        # Собираем сообщение для пользователя
        # Collecting message for user
        msg = TEXTS.get("chain_package_question", "Direct package {0}->{1} not found.\nDo you want to install the chain via English?\nThis may affect translation quality.").format(from_lang, to_lang)
        if need_from:
            msg += f"\n\nWill install {from_lang} → English."
        if need_to:
            msg += f"\nWill install English → {to_lang}."
        answer = messagebox.askyesno(TEXTS.get("question_title", "Confirmation"), msg)
        if not answer:
            return False

        # Устанавливаем только недостающие
        # Install only the missing ones
        if need_from:
            if not self._install_single_package(en_pkg_from, from_lang, "English"):
                return False
        if need_to:
            if not self._install_single_package(en_pkg_to, "English", to_lang):
                return False

        self.translator._refresh_installed_languages()
        return True

    def _install_single_package(self, package, from_lang, to_lang):
        self.progress_overlay.show(TEXTS.get("downloading_package", "Downloading package..."), mode='determinate')
        try:
            self.progress_overlay.update(20, TEXTS.get("downloading_package", "Downloading..."))
            download_path = package.download()
            self.progress_overlay.update(60, TEXTS.get("installing_package", "Installing package..."))
            argostranslate.package.install_from_path(download_path)
            self.progress_overlay.update(100, TEXTS.get("package_install_success", "Package installed"))
            self.progress_overlay.hide()
            messagebox.showinfo(TEXTS.get("info_title", "Information"),
                                TEXTS.get("package_install_success", "Translation package {} -> {} successfully installed.").format(from_lang, to_lang))
            return True
        except Exception as e:
            self.progress_overlay.hide()
            messagebox.showerror(TEXTS.get("error_title", "Error"),
                                 TEXTS.get("package_install_fail", "Failed to install package: {}").format(e))
            return False

    # ========== СКАЧИВАНИЕ ВСЕХ МОДЕЛЕЙ WHISPER / DOWNLOADING ALL WHISPER MODELS ==========
    def download_all_whisper_models(self):
        def task():
            models = ["tiny", "base", "small", "medium", "large"]
            total = len(models)
            for i, model_size in enumerate(models):
                percent = int((i / total) * 100)
                self.progress_overlay.show(TEXTS.get("downloading_whisper", "Downloading Whisper model: {}").format(model_size),
                                           mode='determinate')
                self.progress_overlay.update(percent, f"Downloading {model_size}...")
                recognizer = SpeechRecognizer(model_size=model_size)
                recognizer.load_model()
                del recognizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            self.progress_overlay.hide()
            self.root.after(0, lambda: messagebox.showinfo(TEXTS.get("info_title", "Information"),
                                                           TEXTS.get("downloading_whisper_complete", "All Whisper models have been downloaded.")))
        threading.Thread(target=task, daemon=True).start()

    # ========== СКАЧИВАНИЕ ВСЕХ ПАКЕТОВ ARGOS (ТОЛЬКО ПОДДЕРЖИВАЕМЫЕ ЯЗЫКИ) / DOWNLOADING ALL ARGOS PACKAGES (ONLY SUPPORTED LANGUAGES) ==========
    def download_all_argos_packages(self):
        """Скачивает все языковые пакеты Argos Translate для языков, поддерживаемых программой.
        Downloads all Argos Translate language packages for languages supported by the program."""
        def task():
            try:
                # Получаем список всех доступных пакетов
                # Get list of all available packages
                all_packages = argostranslate.package.get_available_packages()
                # Формируем набор кодов языков, поддерживаемых программой
                # Build a set of language codes supported by the program
                supported_codes = set()
                for lang in self.translator.SUPPORTED_LANGUAGES.keys():
                    code = self.translator.get_language_code(lang, "argos")
                    if code:
                        supported_codes.add(code)
                # Отбираем пакеты, у которых оба языка входят в поддерживаемые
                # Filter packages where both from and to languages are supported
                packages = [pkg for pkg in all_packages 
                        if pkg.from_code in supported_codes and pkg.to_code in supported_codes]
                total = len(packages)
                if total == 0:
                    self.root.after(0, lambda: messagebox.showinfo(
                        TEXTS.get("info_title", "Information"),
                        TEXTS.get("no_packages_available", "No packages available for supported languages.")
                    ))
                    return
                self.progress_overlay.show(
                    TEXTS.get("downloading_argos", "Downloading language packages for supported languages..."),
                    mode='determinate'
                )
                for i, pkg in enumerate(packages):
                    # Пропускаем уже установленные пакеты
                    # Skip already installed packages
                    if pkg in argostranslate.package.get_installed_packages():
                        continue
                    percent = int((i / total) * 100)
                    self.progress_overlay.update(
                        percent,
                        TEXTS.get("downloading_argos_progress", "Downloading {}→{} ({}/{})")
                        .format(pkg.from_code, pkg.to_code, i+1, total)
                    )
                    download_path = pkg.download()
                    argostranslate.package.install_from_path(download_path)
                self.progress_overlay.hide()
                self.root.after(0, lambda: messagebox.showinfo(
                    TEXTS.get("info_title", "Information"),
                    TEXTS.get("downloading_argos_complete", "All available language packages for supported languages have been downloaded.")
                ))
                self.translator._refresh_installed_languages()
            except Exception as e:
                self.progress_overlay.hide()
                self.root.after(0, lambda: messagebox.showerror(TEXTS.get("error_title", "Error"), str(e)))
        threading.Thread(target=task, daemon=True).start()

    # ========== РАСПОЗНАВАНИЕ / RECOGNITION ==========
    def recognize(self):
        input_path = self.stt_input_path.get()
        if not input_path or not os.path.exists(input_path):
            messagebox.showwarning(TEXTS.get("warning_title", "Warning"), TEXTS.get("warning_no_audio_file", "Audio file not found."))
            return

        source_lang = self.stt_source_lang.get()
        target_lang = self.stt_target_lang.get()
        do_translate = self.translate_enabled.get() and (source_lang != target_lang)

        whisper_lang_code = self.translator.get_language_code(source_lang, "whisper")
        if not whisper_lang_code:
            messagebox.showerror(TEXTS.get("error_title", "Error"), f"Source language {source_lang} not supported by Whisper.")
            return

        def task():
            try:
                if self.stt_engine.model is None:
                    self.progress_overlay.show(TEXTS.get("stt_status_loading_model", "Loading {} model...").format(self.stt_engine.model_size), mode='determinate')
                    self.stt_engine.load_model(progress_callback=lambda p, msg: self.progress_overlay.update(p, msg))
                    self.progress_overlay.hide()

                self.stt_status.set(TEXTS.get("stt_status_recognizing", "Recognizing..."))
                original_text = self.stt_engine.recognize(input_path, language_code=whisper_lang_code)
                self.root.after(0, lambda: self._set_recognized_text(original_text))

                if do_translate and original_text.strip():
                    # Проверяем доступность перевода (прямой или через английский)
                    # Checking the availability of translation (direct or through English)
                    if not self.translator.is_translation_available(source_lang, target_lang):
                        self.root.after(0, lambda: self.stt_status.set(TEXTS.get("stt_status_checking_package", "Checking translation package...")))
                        installed = self._install_translation_package(source_lang, target_lang)
                        if not installed:
                            self.root.after(0, lambda: self._set_translated_text(TEXTS.get("stt_fallback_translation_unavailable", "(Translation unavailable)")))
                            self.stt_status.set(TEXTS.get("stt_status_ready", "Ready"))
                            out_path = self.stt_output_path.get()
                            if out_path:
                                with open(out_path, 'w', encoding='utf-8') as f:
                                    f.write(original_text)
                            return
                        # После установки обновляем кэш
                        # Updating cache after installation
                        self.translator._refresh_installed_languages()

                    self.stt_status.set(TEXTS.get("stt_status_translating", "Translating..."))
                    translated = self.translator.translate(original_text, source_lang, target_lang)
                    self.root.after(0, lambda: self._set_translated_text(translated))
                    out_path = self.stt_output_path.get()
                    if out_path:
                        with open(out_path, 'w', encoding='utf-8') as f:
                            f.write(translated)
                else:
                    self.root.after(0, lambda: self._set_translated_text(""))
                    out_path = self.stt_output_path.get()
                    if out_path:
                        with open(out_path, 'w', encoding='utf-8') as f:
                            f.write(original_text)
                self.stt_status.set(TEXTS.get("stt_status_ready", "Ready"))
                self.stt_engine.unload_model()
            except Exception as e:
                self.stt_status.set(TEXTS.get("stt_status_error", "Error"))
                self.root.after(0, lambda: messagebox.showerror(TEXTS.get("error_title", "Error"), str(e)))
                self.root.after(0, self.progress_overlay.hide)

        threading.Thread(target=task, daemon=True).start()

    # ========== СИНТЕЗ / SYNTHESIS ==========
    def synthesize(self):
        if self.synth_mode.get() == "file":
            file_path = self.tts_text_file.get()
            if not file_path or not os.path.exists(file_path):
                messagebox.showwarning(TEXTS.get("warning_title", "Warning"), TEXTS.get("warning_no_text_file", "Text file not found."))
                return
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    original_text = f.read().strip()
            except Exception as e:
                messagebox.showerror(TEXTS.get("error_title", "Error"), TEXTS.get("error_read_file", "Failed to read file: {}").format(e))
                return
        else:
            original_text = self.tts_text_entry.get(1.0, tk.END).strip()
            if not original_text:
                messagebox.showwarning(TEXTS.get("warning_title", "Warning"), TEXTS.get("warning_enter_text", "Enter text to synthesize."))
                return

        ref_path = self.tts_reference_path.get()
        if not ref_path or not os.path.exists(ref_path):
            messagebox.showwarning(TEXTS.get("warning_title", "Warning"), TEXTS.get("warning_no_reference", "Voice sample not found."))
            return

        out_path = self.tts_output_path.get()
        if not out_path:
            timestamp = int(time.time())
            out_path = os.path.join(OUTPUT_DIR, TEXTS.get("default_output_wav", "output_{}.wav").format(timestamp))
            self.tts_output_path.set(out_path)

        source_lang = self.tts_source_lang.get()
        target_lang = self.tts_target_lang.get()
        need_translation = (source_lang != target_lang)

        final_text = original_text

        def task():
            nonlocal final_text
            try:
                if need_translation and original_text.strip():
                    # Проверяем доступность перевода (прямой или через английский)
                    if not self.translator.is_translation_available(source_lang, target_lang):
                        installed = self._install_translation_package(source_lang, target_lang)
                        if not installed:
                            self.tts_status.set(TEXTS.get("tts_status_error", "Error"))
                            return
                        self.translator._refresh_installed_languages()
                    self.tts_status.set(TEXTS.get("tts_status_translating", "Translating text..."))
                    translated = self.translator.translate(original_text, source_lang, target_lang)
                    final_text = translated
                    self.tts_status.set(TEXTS.get("tts_status_synthesizing", "Synthesizing..."))

                self.tts_status.set(TEXTS.get("tts_status_synthesizing", "Synthesizing..."))
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                if os.path.exists(out_path):
                    try:
                        os.remove(out_path)
                    except Exception:
                        pass
                tts_lang_code = self.translator.get_language_code(target_lang, "tts")
                result = self.tts_engine.clone_voice(final_text, ref_path, out_path, language=tts_lang_code)
                if result:
                    self.last_synthesized.set(result)
                    self.root.after(0, lambda: self.play_btn.config(state=tk.NORMAL))
                    self.root.after(0, lambda: self.stop_btn.config(state=tk.NORMAL))
                    self.root.after(0, lambda: self.player_file_label.config(text=os.path.basename(result)))
                    self.tts_status.set(TEXTS.get("tts_status_ready", "Ready"))
                else:
                    self.tts_status.set(TEXTS.get("tts_status_error", "Error"))
            except Exception as e:
                self.tts_status.set(TEXTS.get("tts_status_error", "Error"))
                self.root.after(0, lambda: messagebox.showerror(TEXTS.get("error_title", "Error"), str(e)))

        threading.Thread(target=task, daemon=True).start()

    def _set_recognized_text(self, text):
        self.recognized_text.config(state=tk.NORMAL)
        self.recognized_text.delete(1.0, tk.END)
        self.recognized_text.insert(1.0, text)
        self.recognized_text.config(state=tk.DISABLED)

    def _set_translated_text(self, text):
        self.translated_text.config(state=tk.NORMAL)
        self.translated_text.delete(1.0, tk.END)
        self.translated_text.insert(1.0, text)
        self.translated_text.config(state=tk.DISABLED)

    def play_audio(self):
        file = self.last_synthesized.get()
        if not file or not os.path.exists(file):
            messagebox.showinfo(TEXTS.get("info_title", "Information"), TEXTS.get("warning_no_file_to_play", "No file to play."))
            return
        try:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
            pygame.mixer.music.load(file)
            pygame.mixer.music.play()
        except Exception as e:
            messagebox.showerror(TEXTS.get("error_title", "Error"), TEXTS.get("error_play_audio", "Failed to play: {}").format(e))

    def stop_audio(self):
        pygame.mixer.music.stop()

# ============================================================
# ЗАПУСК / LAUNCH
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
