import requests
import json

def emotion_detector(text_to_analyze):
    model_url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    header = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    user_input = {"raw_document": {"text": text_to_analyze}}
    response = requests.post(model_url, json = user_input, headers = header)
    formatted_response = json.loads(response.text)

    emotions_obj = formatted_response['emotionPredictions'][0]['emotion']
    emotions = {
        'anger': emotions_obj['anger'],
        'disgust': emotions_obj['disgust'],
        'fear': emotions_obj['fear'],
        'joy': emotions_obj['joy'],
        'sadness': emotions_obj['sadness'],
        'dominant_emotion': max(zip(emotions_obj.values(), emotions_obj.keys()))[1] 
    }

    return emotions
