import io
from pydantic import BaseModel

class GenerateTextInferenceRequest(BaseModel):
  audio : bytes

class GenerateTextInferenceResponse(BaseModel):
  text : str