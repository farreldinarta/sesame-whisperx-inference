import io
import os
import base64
import torch
import torchaudio
import tempfile
from app.utils.object import singleton
from app.provider.llm.interface import LLMInterface
from app.provider.llm.sesame_whisperx_utils.transcribe import WhisperModel 
from app.configs.environment import get_environment_variables

env = get_environment_variables()
@singleton
class WhisperXLLMProvider(LLMInterface):
  __device : str
  __compute_type : str
  __model_type : str

  def __init__(self):

    if torch.backends.mps.is_available():
        self.__device = "mps"
        self.__compute_type = "float16"
        self.__model_type = "large-v3"
    elif torch.cuda.is_available():
        self.__device = "cuda"
        self.__compute_type = "float16"
        self.__model_type = "large-v3"
    else:
        self.__device = "cpu"
        self.__compute_type = "int8"
        self.__model_type = "small"

    print("Used Device : ", self.__device)

    self.__model = WhisperModel(self.__model_type, device=self.__device, compute_type=self.__compute_type)
    
  async def inference(self, audio) -> list:
    segments, _= self.__model.transcribe(
        audio, 
        beam_size=5, 
        language="en", 
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),        
    )
    return segments
  
def get_whisperx_model() -> LLMInterface:
  return WhisperXLLMProvider()
