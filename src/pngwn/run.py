import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

def load_model():
    base_id = "Qwen/Qwen3.5-4B-Base"
    peft_id = "pngwn/system-one-qwen3.5-4b-scorer"
    
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(peft_id)
    # Ensure pad token is set for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load base model with a single sequence classification head (scalar output)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_id,
        num_labels=1,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(base_model.config, "get_text_config"):
        base_model.config.get_text_config().pad_token_id = tokenizer.pad_token_id
    
    # Attach the Jev-style LoRA adapter
    model = PeftModel.from_pretrained(base_model, peft_id)
    model.eval()
    return tokenizer, model

def score_decision(tokenizer, model, state, question, options):
    """
    Evaluates all options in parallel without autoregressive text generation.
    """
    # 1. Construct the (state, question, option) triple for every option
    inputs_text = []
    for option in options:
        # If the adapter uses a specific chat template, apply it. Otherwise, construct a clean text prompt.
        prompt = f"Context: {state}\nQuestion: {question}\nOption: {option}"
        inputs_text.append(prompt)
        
    # 2. Tokenize the batch of options. 
    # Note: The model card notes a 384-token truncation limit for this specific adapter.
    encoded = tokenizer(
        inputs_text, 
        padding=True, 
        truncation=True, 
        max_length=384, 
        return_tensors="pt"
    ).to(model.device)
    
    # 3. Forward Pass
    with torch.no_grad():
        # Outputs shape is [num_options, 1]
        outputs = model(**encoded)
        logits = outputs.logits.squeeze(-1) # Flatten to [num_options]
        
        # Temperature of 1.75 was fitted on the validation split for optimal calibration
        temperature = 1.75 
        
        # Softmax across the batch dimension to get a valid probability distribution
        probabilities = F.softmax(logits / temperature, dim=0)
        
    # Combine options with their probabilities and sort descending
    results = {opt: prob.item() for opt, prob in zip(options, probabilities)}
    return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

if __name__ == "__main__":

    tokenizer, model = load_model()
    
    # Define an unstructured state/context
    state_context = (
        "I reset my password this morning because the app told me it had expired. "
        "Since then every login attempt fails and now the account says it is locked after "
        "three failed attempts. I need to run the payroll batch by 4pm."
    )
    
    # Define a strongly typed question
    question_text = "Which department should handle this ticket?"
    
    # Define exactly the options you want to force the model into
    candidate_options = [
        "billing",
        "technical support",
        "sales",
        "security",
        "general"
    ]
    
    print("\nScoring options...")
    results = score_decision(tokenizer, model, state_context, question_text, candidate_options)
    
    print("\n=== Calibrated Decision ===")
    for opt, prob in results.items():
        print(f"{opt.ljust(20)} : {prob:.1%}")