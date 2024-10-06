import requests

def emotion_detector(text_to_analyze):
    model_url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    header = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    user_input = {"raw_document": {"text": text_to_analyze}}

    response = requests.post(model_url, json = user_input, headers = header)
    return response.text
