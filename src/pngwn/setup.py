import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

def setup():
    print("\nDownloading and caching models (this may take a few minutes)...")

    base_id = "Qwen/Qwen3.5-4B-Base"
    peft_id = "pngwn/system-one-qwen3.5-4b-scorer"

    # Download tokenizer from the adapter repo
    AutoTokenizer.from_pretrained(peft_id) 

    # Download base model (Note: num_labels=1 attaches a scalar scoring head)
    # Using torch_dtype=torch.bfloat16 ensures the model fits in memory and prevents
    # accelerate from offloading parameters to disk/meta device.
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_id, 
        num_labels=1, 
        dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Download the PEFT LoRA adapter
    PeftModel.from_pretrained(base_model, peft_id)
    
    print(f"\nSetup complete! The {peft_id} model is ready for use.")

if __name__ == "__main__":
    setup()
