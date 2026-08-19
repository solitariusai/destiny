import requests

url = "https://huggingface.co/google/gemma-4-26B-A4B-it/raw/main/config.json"
response = requests.get(url)
if response.status_code == 200:
    print(response.text)
else:
    print("Failed to get config:", response.status_code)
