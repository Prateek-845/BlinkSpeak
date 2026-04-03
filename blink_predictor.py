import os
import re
import urllib.request
import torch
import transformers 
from transformers import AutoModelForCausalLM, AutoTokenizer

transformers.logging.set_verbosity_error()

class SmartPredictor:
    def __init__(self):
        self.model_name = "gpt2-large"
        self.dict_file = "common_words.txt"
        self.valid_words = []
        self._load_dictionary()
        self._load_model()

    def _load_dictionary(self):
        if not os.path.exists(self.dict_file):
            url = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-no-swears.txt"
            urllib.request.urlretrieve(url, self.dict_file)

        with open(self.dict_file, 'r') as f:
            self.valid_words = [w.strip() for w in f.readlines() if len(w.strip()) > 1]

    def _load_model(self):
        print(f"Loading {self.model_name} into memory")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            if torch.cuda.is_available():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, load_in_8bit=True, device_map="auto")
            else:
                raise ValueError("CUDA not available.")
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.to(self.device)

    def _generate_ai_prediction(self, text_context):
        inputs = self.tokenizer(text_context, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]
        
        with torch.no_grad(): 
            outputs = self.model.generate(
                **inputs, max_new_tokens=5, num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id, do_sample=True, 
                temperature=0.3, top_p=0.9, repetition_penalty=1.15,
                no_repeat_ngram_size=2, use_cache=True 
            )

        new_tokens = outputs[0][input_length:]
        new_prediction = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        starts_with_space = new_prediction.startswith(' ')
        temp_pred = re.sub(r'[^a-zA-Z0-9.,!?\'" \-]', '', new_prediction.lstrip())
        
        if not temp_pred:
            return ""
        
        temp_pred = re.split(r'[ .,!?\n]', temp_pred)[0]    
        return (' ' if starts_with_space else '') + temp_pred

    def get_suggestion(self, text_context):
        if not text_context:
            return ""

        match = re.search(r'([a-zA-Z]+)$', text_context)
        
        if match:
            partial_word = match.group(1).lower()
            if partial_word in ["my", "me", "is", "to", "do", "he", "we", "be", "it", "in", "on", "as", "at"]:
                return ""

            if len(partial_word) >= 2:
                matches = [w for w in self.valid_words if w.startswith(partial_word) and w != partial_word][:5]
                if matches:
                    return sorted(matches, key=len)[0][len(partial_word):]
            return "" 
        else:
            words = text_context.split()
            if not words:
                return ""
            
            short_context = words[-1] + " "
            return self._generate_ai_prediction(short_context)