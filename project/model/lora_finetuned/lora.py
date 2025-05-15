import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class LoRALayer(nn.Module):
    def __init__(self, orig_layer, r=4, alpha=32):
        super().__init__()
        self.orig_layer = orig_layer  # Original weight matrix (frozen)
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # LoRA matrices, initialized randomly
        self.lora_A = nn.Parameter(torch.randn(orig_layer.weight.size(0), r) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(r, orig_layer.weight.size(1)) * 0.01)
        
        # Freeze original weight matrix
        self.orig_layer.weight.requires_grad = False

    def forward(self, x):
        # Original projection
        result = self.orig_layer(x)
        
        # LoRA adaptation
        lora_update = (x @ self.lora_B.t()) @ self.lora_A.t() * self.scaling
        return result + lora_update

def apply_lora_to_model(model, r=4, alpha=32):
    """
    Replace projection layers in model's attention modules with LoRA-augmented layers.
    This example targets the 'q_proj', 'k_proj', and 'v_proj' linear layers in GPT-like models.
    """
    for name, module in model.named_modules():
        # Example for GPT-like models with 'q_proj', 'k_proj', 'v_proj' Linear layers in attention
        if isinstance(module, nn.Linear) and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name):
            parent = model
            *path, attr = name.split('.')
            for p in path:
                parent = getattr(parent, p)
            orig_layer = getattr(parent, attr)
            lora_layer = LoRALayer(orig_layer, r=r, alpha=alpha)
            setattr(parent, attr, lora_layer)
    return model

# Example usage:

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Apply LoRA adapters to the model
model = apply_lora_to_model(model, r=4, alpha=32)

# Now model parameters will be mostly frozen except LoRA params
print("Trainable parameters:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.shape)

# Example training loop skeleton (pseudo-code)
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
# for batch in dataloader:
#     inputs = tokenizer(batch['text'], return_tensors='pt', padding=True)
#     outputs = model(**inputs, labels=inputs["input_ids"])
#     loss = outputs.loss
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
