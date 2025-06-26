from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=api_key)

def gemini_2p0_flash_process(video_path, question):
    with open(video_path, 'rb') as f:
        video_bytes = f.read()
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=types.Content(
            parts=[
                types.Part(
                    inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')
                ),
                types.Part(text=question)
            ]
        )
    )

    return response.text