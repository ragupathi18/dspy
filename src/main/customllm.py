from dsp import LM
import requests

class Claude(LM):
    def __init__(self,model, api_key):
        super().__init__(model)
        self.model = model
        self.api_key = "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"
        self.provider = "default"

        self.base_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

    def basic_request(self, prompt: str, **kwargs):
        headers = {"Authorization": f"Bearer {self.api_key}",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "messages-2023-12-15",
            "content-type": "application/json"
        }

        data = {
            **kwargs,
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        data={
            **kwargs,
            "inputs":prompt
        }
        response = requests.post(self.base_url, headers=headers, json=data)
        response = response.json()

        self.history.append({
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
        })
        return response

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        response = self.request(prompt, **kwargs)

        completions = [result["generated_text"] for result in response]

        return completions