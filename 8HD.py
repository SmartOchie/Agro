#!/usr/bin/env python
# coding: utf-8
# %%

# ## Cognitive Services - Speech to Text

# %%


import azure.cognitiveservices.speech as speechsdk

def transcribe_audio(audio_file_path):
    speech_config = speechsdk.SpeechConfig(subscription="2ILmHcBFSLKFf7eKDslVRbm2k12cLjRWAY7agiJIWk6betvlJvYWJQQJ99BDACYeBjFXJ3w3AAAYACOGEW3N", region="eastus")
    audio_input = speechsdk.AudioConfig(filename=audio_file_path)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config, audio_input)
    result = speech_recognizer.recognize_once()
    return result.text


# ## Cognitive Service - Image to Text

# %%


from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import time

def extract_text_from_image(image_path):
    client = ComputerVisionClient(
        endpoint="https://smart-agro-vision-service.cognitiveservices.azure.com/",
        credentials=CognitiveServicesCredentials("1vQKpp1KbnEKt4Tj9WLikvgXhZFlWvUWfx2l0UyPq6OD0MStOsCnJQQJ99BDACYeBjFXJ3w3AAAFACOGxZ7L")
    )
    with open(image_path, "rb") as image_stream:
        read_response = client.read_in_stream(image_stream, raw=True)
    operation_location = read_response.headers["Operation-Location"]
    operation_id = operation_location.split("/")[-1]
    
    while True:
        result = client.get_read_result(operation_id)
        if result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)

    return " ".join([line.text for r in result.analyze_result.read_results for line in r.lines])


# ## Custom Logic to Enhance Recommendation

# %%


def enrich_with_agro_advice(user_input, gpt_response):
    """
    Adds agro-specific advice to GPT output based on user input.
    """
    user_input = user_input.lower()

    additional_info = ""

    if "maize" in user_input and "yellow" in user_input:
        additional_info = (
            "💡 Agro Tip: Yellowing maize leaves often indicate nitrogen deficiency. "
            "Apply urea or compost for better results."
            "Recommended: Apply NPK 20:10:10 fertilizer at planting."
            "disease: Check for fungal spots; apply Mancozeb if present"
        )

    elif "cassava" in user_input and "stunted" in user_input:
        additional_info = (
            "💡 Agro Tip: Stunted cassava growth may be due to poor soil fertility. "
            "Consider using NPK fertilizer and ensure good soil drainage."
        )

    elif "cowpea" in user_input and "insect" in user_input:
        additional_info = (
            "💡 Agro Tip: Use neem extract or approved insecticides to control cowpea pests. "
            "Rotate crops to prevent infestation."
        )

       # If no agro advice matched, just return GPT response
    if additional_info:
        return gpt_response + "\n\n" + additional_info
    else:
        return gpt_response


# ## Full Bot Logic in Flask

# %%


import os
from flask import Flask, request, Response
from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings, TurnContext
from botbuilder.schema import Activity
from dotenv import load_dotenv
import openai

from speech_to_text import transcribe_audio
from image_to_text import extract_text_from_image
from custom_logic import enhance_recommendation

load_dotenv()

app = Flask(__name__)

# Adapter setup
adapter_settings = BotFrameworkAdapterSettings(os.getenv("BOT_APP_ID"), os.getenv("BOT_APP_PASSWORD"))
adapter = BotFrameworkAdapter(adapter_settings)

# OpenAI config
openai.api_key = os.getenv("OPENAI_API_KEY")

async def process_input(turn_context: TurnContext):
    user_message = turn_context.activity.text

    # Basic multimodal trigger keywords
    if user_message.startswith("audio:"):
        audio_path = user_message.replace("audio:", "").strip()
        user_message = transcribe_audio(audio_path)
    elif user_message.startswith("image:"):
        image_path = user_message.replace("image:", "").strip()
        user_message = extract_text_from_image(image_path)

    # Enhance logic before GPT
    custom_reply = enhance_recommendation(user_message)
    if custom_reply:
        await turn_context.send_activity(custom_reply)
        return

    # GPT fallback
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": user_message}]
    )
    bot_reply = response["choices"][0]["message"]["content"]
    await turn_context.send_activity(bot_reply)

@app.route("/api/messages", methods=["POST"])
def messages():
    if "application/json" in request.headers["Content-Type"]:
        body = request.json
    else:
        return Response(status=415)

    activity = Activity().deserialize(body)
    auth_header = request.headers.get("Authorization", "")

    import asyncio
    asyncio.run(adapter.process_activity(activity, auth_header, process_input))
    return Response(status=200)

if __name__ == '__main__':
    app.run()


