import io
import os
import base64
import torch
import torchaudio
import tempfile
from app.utils.object import singleton
from app.provider.llm.interface import LLMInterface
from app.provider.llm.sesame_whisperx_streaming_utils.whisper_online import FasterWhisperASR, OnlineASRProcessor
from app.configs.environment import get_environment_variables

env = get_environment_variables()
@singleton
class WhisperXStreamLLMProvider(LLMInterface):
  __device : str
  __compute_type : str
  __model_type : str

  def __init__(self):

    if torch.backends.mps.is_available():
        self.__device = "mps"
        self.__compute_type = "float16"
        self.__model_type = "large-v2"
    elif torch.cuda.is_available():
        self.__device = "cuda"
        self.__compute_type = "float16"
        self.__model_type = "large-v2"
    else:
        self.__device = "cpu"
        self.__compute_type = "int8"
        self.__model_type = "small"

    print("Used Device : ", self.__device)

    self.__model = FasterWhisperASR("en", "large-v2")
    self.__asr_processor = OnlineASRProcessor(self.__model)

  def get_asr_processor(self):
    return self.__asr_processor
    
  async def inference(self, audio) -> str:
    self.__asr_processor.insert_audio_chunk(audio)
    partial_output = self.__asr_processor.process_iter()
    return partial_output

  
def get_whisperx_model() -> LLMInterface:
  return WhisperXStreamLLMProvider()
