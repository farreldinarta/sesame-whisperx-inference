from fastapi import Depends
from app.utils.object import singleton
from app.provider.llm.interface import LLMInterface
from app.provider.llm.sesame_whisperx_stream_llm_provider import get_whisperx_model

@singleton
class LLM:
  def __init__(self, whisperx: LLMInterface):
    self.__whisperx = whisperx

  def get_whisperx_model(self) -> LLMInterface:
    return self.__whisperx
  

def get_llm(whisperx : LLMInterface = Depends(get_whisperx_model)) -> LLM:
  return LLM(whisperx=whisperx)