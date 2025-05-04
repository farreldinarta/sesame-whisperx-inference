import io
from pydantic import BaseModel

class GenerateTextInferenceRequest(BaseModel):
  audio : io.BytesIO

class GenerateTextInferenceResponse(BaseModel):
  text : str