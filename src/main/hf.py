# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests

API_URL =  "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
#"https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "who is the president of the USA?",
})

print(output)
#tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
#model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")



