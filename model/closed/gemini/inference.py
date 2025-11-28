from google import genai
from google.genai import types
import os, time
from dotenv import load_dotenv

load_dotenv()

def gemini_process(media_path, question, model_name, idx):
    # choose API key shard
    # api_key = os.getenv(f'GEMINI_API_KEY{idx % 8}')
    api_key = os.getenv(f'KEY')
    client = genai.Client(api_key=api_key)

    # decide mime type by file extension
    ext = os.path.splitext(media_path)[1].lower()
    mime = 'audio/wav' if ext == '.wav' else 'video/mp4'

    print(media_path, mime)

    with open(media_path, 'rb') as f:
        payload = f.read()

    while True:
        try:
            if "pro" in model_name or "2.5" not in model_name:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=types.Content(parts=[
                        types.Part(inline_data=types.Blob(data=payload, mime_type=mime)),
                        types.Part(text=question),
                    ])
                )
            else:
                print(f"{model_name} not thinking")
                resp = client.models.generate_content(
                    model=model_name,
                    contents=types.Content(parts=[
                        types.Part(inline_data=types.Blob(data=payload, mime_type=mime)),
                        types.Part(text=question),
                    ]),
                    config = types.GenerateContentConfig(
                        thinking_config = types.ThinkingConfig(thinking_budget = 0)
                    )
                )
            return resp.text
        except Exception as e:
            print(f"Got exception: {e}, retry again.")
            time.sleep(40)
