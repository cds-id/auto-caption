"""
Whisper model management and utilities.
"""

import os
from typing import Dict, List, Any, Optional
from pathlib import Path

import whisper
import torch


# Model specifications
WHISPER_MODELS = {
    "tiny": {
        "name": "tiny",
        "parameters": "39M",
        "english_only": True,
        "multilingual": True,
        "vram": "~1 GB",
        "speed": "~32x",
        "relative_speed": 32
    },
    "base": {
        "name": "base",
        "parameters": "74M",
        "english_only": True,
        "multilingual": True,
        "vram": "~1 GB",
        "speed": "~16x",
        "relative_speed": 16
    },
    "small": {
        "name": "small",
        "parameters": "244M",
        "english_only": True,
        "multilingual": True,
        "vram": "~2 GB",
        "speed": "~6x",
        "relative_speed": 6
    },
    "medium": {
        "name": "medium",
        "parameters": "769M",
        "english_only": True,
        "multilingual": True,
        "vram": "~5 GB",
        "speed": "~2x",
        "relative_speed": 2
    },
    "large": {
        "name": "large",
        "parameters": "1550M",
        "english_only": False,
        "multilingual": True,
        "vram": "~10 GB",
        "speed": "1x",
        "relative_speed": 1
    }
}


def get_available_models() -> List[Dict[str, Any]]:
    """
    Get list of available Whisper models with their specifications.
    
    Returns:
        List of model dictionaries with specifications
    """
    return list(WHISPER_MODELS.values())


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary with model specifications
        
    Raises:
        ValueError: If model name is not valid
    """
    if model_name not in WHISPER_MODELS:
        raise ValueError(f"Invalid model name: {model_name}. "
                        f"Available models: {', '.join(WHISPER_MODELS.keys())}")
    
    return WHISPER_MODELS[model_name]


def get_model_path(model_name: str) -> Path:
    """
    Get the local path where a model is stored.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Path to the model file
    """
    # Get the default Whisper model directory
    default_dir = os.path.join(os.path.expanduser("~"), ".cache", "whisper")
    model_file = f"{model_name}.pt"
    
    return Path(default_dir) / model_file


def is_model_downloaded(model_name: str) -> bool:
    """
    Check if a model is already downloaded.
    
    Args:
        model_name: Name of the model
        
    Returns:
        True if model is downloaded, False otherwise
    """
    model_path = get_model_path(model_name)
    return model_path.exists()


def get_device() -> str:
    """
    Get the best available device for running models.
    
    Returns:
        Device string ('cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available compute devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device": get_device()
    }
    
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_allocated"] = torch.cuda.memory_allocated(0)
        info["cuda_memory_reserved"] = torch.cuda.memory_reserved(0)
    
    return info


class WhisperModel:
    """
    Wrapper class for Whisper models with additional functionality.
    """
    
    def __init__(self, model_name: str = "base", device: Optional[str] = None):
        """
        Initialize a Whisper model wrapper.
        
        Args:
            model_name: Name of the model to use
            device: Device to use ('cuda' or 'cpu', None for auto)
        """
        self.model_name = model_name
        self.model_info = get_model_info(model_name)
        self.device = device or get_device()
        self.model = None
    
    def download(self, force: bool = False) -> bool:
        """
        Download the model if not already present.
        
        Args:
            force: Force re-download even if model exists
            
        Returns:
            True if download was performed, False if already existed
        """
        if not force and is_model_downloaded(self.model_name):
            return False
        
        # Whisper's load_model will download if needed
        whisper.load_model(self.model_name, device="cpu", download_root=None)
        return True
    
    def load(self):
        """Load the model into memory."""
        if self.model is None:
            self.model = whisper.load_model(self.model_name, device=self.device)
    
    def unload(self):
        """Unload the model from memory."""
        self.model = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
    
    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio using the model.
        
        Args:
            audio_path: Path to audio file
            **kwargs: Additional arguments for transcription
            
        Returns:
            Transcription results
        """
        if self.model is None:
            self.load()
        
        return self.model.transcribe(audio_path, **kwargs)
    
    def get_memory_usage(self) -> Dict[str, int]:
        """
        Get current memory usage of the model.
        
        Returns:
            Dictionary with memory usage information
        """
        if self.device == "cuda" and self.model is not None:
            return {
                "allocated": torch.cuda.memory_allocated(0),
                "reserved": torch.cuda.memory_reserved(0)
            }
        else:
            return {
                "allocated": 0,
                "reserved": 0
            }
    
    def estimate_processing_time(self, audio_duration: float) -> float:
        """
        Estimate processing time for audio of given duration.
        
        Args:
            audio_duration: Duration of audio in seconds
            
        Returns:
            Estimated processing time in seconds
        """
        # Base estimate on relative speed
        relative_speed = self.model_info["relative_speed"]
        
        # Assume "large" model processes at real-time on average hardware
        base_time = audio_duration
        
        # Adjust for model speed
        estimated_time = base_time / relative_speed
        
        # Add overhead for loading, preprocessing, etc.
        overhead = 5.0  # seconds
        
        return estimated_time + overhead


def get_recommended_model(
    audio_duration: float,
    available_memory: Optional[int] = None,
    quality_preference: str = "balanced"
) -> str:
    """
    Get recommended model based on constraints and preferences.
    
    Args:
        audio_duration: Duration of audio in seconds
        available_memory: Available GPU memory in GB (None for auto-detect)
        quality_preference: 'speed', 'balanced', or 'quality'
        
    Returns:
        Recommended model name
    """
    # Auto-detect available memory if not provided
    if available_memory is None:
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            available_memory = total_memory / (1024**3)  # Convert to GB
        else:
            # Assume CPU with reasonable RAM
            available_memory = 8.0
    
    # Define recommendations based on preferences
    if quality_preference == "speed":
        # Prefer faster models
        if available_memory >= 1:
            return "tiny"
        else:
            return "tiny"
    
    elif quality_preference == "quality":
        # Prefer larger models that fit in memory
        if available_memory >= 10:
            return "large"
        elif available_memory >= 5:
            return "medium"
        elif available_memory >= 2:
            return "small"
        else:
            return "base"
    
    else:  # balanced
        # Balance between speed and quality
        if audio_duration < 300:  # Less than 5 minutes
            if available_memory >= 5:
                return "medium"
            elif available_memory >= 2:
                return "small"
            else:
                return "base"
        else:  # Longer audio
            if available_memory >= 2:
                return "small"
            else:
                return "base"


# Language codes supported by Whisper
WHISPER_LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
}


def get_language_name(code: str) -> str:
    """
    Get full language name from language code.
    
    Args:
        code: Two-letter language code
        
    Returns:
        Full language name
    """
    return WHISPER_LANGUAGES.get(code, code)


def is_valid_language(code: str) -> bool:
    """
    Check if a language code is valid for Whisper.
    
    Args:
        code: Language code to check
        
    Returns:
        True if valid, False otherwise
    """
    return code in WHISPER_LANGUAGES