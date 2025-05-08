import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.dto.inference_dto import GenerateTextInferenceRequest
from app.provider.llm.base import get_llm
from app.provider.llm.sesame_whisperx_stream_llm_provider import WhisperXStreamLLMProvider
from app.service.inference_service import get_inference_service

router = APIRouter()

@router.websocket("")
async def generate(websocket: WebSocket):
    await websocket.accept()

    whisperx_model = WhisperXStreamLLMProvider()
    llm = get_llm(whisperx=whisperx_model)
    llm.get_whisperx_model().get_asr_processor().init()
    service = get_inference_service(llm=llm)

    audio_buffer = b""

    try:
        while True:
            try:
                # Wait up to 3 seconds for the next message
                message = await asyncio.wait_for(websocket.receive(), timeout=3.0)

                if "bytes" in message:
                    audio_buffer += message["bytes"]

                    # Optional: intermediate result when buffer > threshold
                    result = await service.generate(GenerateTextInferenceRequest(audio=audio_buffer))
                    print("whisperX - partial result: ", result.text)   
                    await websocket.send_text(result.text)
                    audio_buffer = b""

                elif "text" in message:
                    if message["text"] == "__END__":
                        if audio_buffer:
                            result = await service.generate(GenerateTextInferenceRequest(audio=audio_buffer))
                            print("whisperX - final result: ", result.text)
                            await websocket.send_text(result.text)
                        llm.get_whisperx_model().get_asr_processor().finish()
                        await websocket.close()
                        break

            except asyncio.TimeoutError:
                # No data received for 3 seconds
                if audio_buffer:
                    result = await service.generate(GenerateTextInferenceRequest(audio=audio_buffer))
                    print("whisperX - final result (timeout): ", result.text)
                    await websocket.send_text(result.text)
                llm.get_whisperx_model().get_asr_processor().finish()
                await websocket.close()
                break

    except WebSocketDisconnect as e:
        print("WebSocket disconnected: ", e)
