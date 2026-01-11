import torch
from ts.torch_handler.base_handler import BaseHandler
from transformers import AutoTokenizer

class HRVLLMHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.tokenizer = None

    def initialize(self, context):
        # Загрузка модели
        print('model')
        model_dir = context.system_properties.get("model_dir")
        self.model = torch.jit.load(f"{model_dir}/model.pt")
        self.model.eval()
        print('yxxxx')

        # Загрузка токенайзера
        print('tokenizer')
        self.tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}/tokenizer")
        self.device = torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print('yxxx')

    def preprocess(self, data):
        input_text = data[0].get("body").get("text", "")
        input_ids = torch.tensor(self.tokenizer.encode(input_text), device=self.device)[None, :]
        return input_ids

    def inference(self, inputs):
        with torch.no_grad():
            model_output = self.model.generate(
                inputs, max_new_tokens=20, eos_token_id=self.tokenizer.eos_token_id, do_sample=True, top_k=10
            )
        return model_output

    def postprocess(self, outputs):
        pred = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [{"prediction": pred}]
