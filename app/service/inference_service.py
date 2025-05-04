import io
from fastapi import Depends
from app.dto.inference_dto import GenerateTextInferenceRequest, GenerateTextInferenceResponse
from app.provider.llm.base import LLM, get_llm
from app.exception.http_exception import WebSocketAPIException
from app.utils.object import singleton

@singleton
class InferenceService:
  def __init__(self, llm: LLM):
    self.__llm = llm

  def segments_to_string(self, segments : list) -> str:
    output_lines = []

    for segment in segments:
        for word in segment.words:
            output_lines.append("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))

    return "\n".join(output_lines)

  async def generate(self, payload : GenerateTextInferenceRequest) -> GenerateTextInferenceResponse:
    segments : list = await self.__llm.get_whisperx_model().inference(payload.audio)
    if not segments:
      raise WebSocketAPIException(
        status=400,
        message="Failed to transcribe audio",
        error="Failed to transcribe audio"
      )
    
    merged_text = self.segments_to_string(segments)
    return GenerateTextInferenceResponse(text=merged_text)

def get_inference_service(llm : LLM = Depends(get_llm)) -> InferenceService:
  return InferenceService(llm=llm)