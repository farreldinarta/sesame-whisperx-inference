from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.dto.inference_dto import GenerateTextInferenceRequest
from app.provider.llm.base import get_llm
from app.provider.llm.sesame_whisperx_llm_provider import WhisperXLLMProvider
from app.service.inference_service import get_inference_service

router = APIRouter()

@router.websocket("")
async def generate(websocket: WebSocket):
    await websocket.accept()

    whisperx_model = WhisperXLLMProvider()
    llm = get_llm(whisperx=whisperx_model)
    service = get_inference_service(llm=llm) 

    audio_buffer = b""
  
    try:
        while True:
            chunk = await websocket.receive_bytes() 
            audio_buffer += chunk

            if len(audio_buffer) > 32000:
                result = await service.generate(GenerateTextInferenceRequest(audio=audio_buffer))
                await websocket.send_text(result.text)
                audio_buffer = b""  

    except WebSocketDisconnect as e:
        print("WebSocket disconnected : ", e)
