from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from openai import OpenAI
from src.db.database import create_message as db_create_message
import json

router = APIRouter()
client = OpenAI()

@router.websocket("/ws/speech-to-text/{started_case_id}")
async def websocket_endpoint(websocket: WebSocket, started_case_id: str):
    try:
        await websocket.accept()
        print(f"WebSocket connection accepted for case ID: {started_case_id}")
        
        while True:
            print("Waiting for audio data...")
            audio_data = await websocket.receive_bytes()
            print(f"Received audio data of size: {len(audio_data)} bytes")
            
            print("Writing audio data to temporary file...")
            temp_file_path = "/tmp/temp_audio.mp3"
            with open(temp_file_path, "wb") as audio_file:
                audio_file.write(audio_data)
            
            print("Starting transcription with Whisper...")
            with open(temp_file_path, "rb") as audio_file:
                transcription = client.audio.translations.create(
                    model="whisper-1",
                    file=audio_file
                )
            print(f"Transcription completed: {transcription.text}")
            
            print("Saving message to database...")
            await db_create_message(websocket.app, {
                "started_case_id": started_case_id,
                "content": transcription.text,
                "is_user_message": True,
                "awaiting_user_input": False
            })
            print("Message saved successfully")
            
            print("Sending transcription back to client...")
            await websocket.send_text(json.dumps({
                "text": transcription.text
            }))
            print("Response sent to client")
            
    except WebSocketDisconnect:
        print("WebSocket disconnected")  
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        try:
            print("Attempting to send error message to client...")
            await websocket.send_text(json.dumps({
                "error": str(e)
            }))
            print("Error message sent to client")
        except:
            print("Failed to send error message to client")