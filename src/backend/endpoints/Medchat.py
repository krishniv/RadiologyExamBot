from transformers import AutoModelForCausalLM, AutoTokenizer

class MedicalChatbot:
    def __init__(self, model_name="microsoft/DialoGPT-small"):
        # Initialize the Hugging Face model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_response(self, user_message):
        """
        Generate a response for a given user message.
        """
        
        inputs = self.tokenizer.encode(user_message, return_tensors="pt")
        outputs = self.model.generate(inputs, max_length=100, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)
        return response


Medchat = MedicalChatbot()