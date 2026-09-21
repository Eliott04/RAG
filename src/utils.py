import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM,AutoTokenizer

def get_best_device():
    """Returns the best device on this computer"""

    if torch.cuda.is_available():
        device = torch.device("cuda")
        total_memory = torch.cuda.get_device_properties(device).total_memory
        print(f"GPU Memory: {total_memory / 1e9:.1f} GB")
        print(f"GPU Name: {torch.cuda.get_device_name(device)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Found device: {device}")
    return device


def get_model(model_name, device, is_splade=False):
    '''
    Docstring for get_model
    
    :param model_name: model to be loaded
    :param device: device to load the model on
    :return: model, tokenizer
    '''
    
    # different model options:
    # 1. SmolLM2-1.7B float16 (~3.4GB) - default, works on all platforms
    # model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    #
    # 2. Pre-quantized models (GPTQ/AWQ) - Linux only, requires:
    #    uv pip install auto-gptq autoawq  (or: uv sync --extra quantized)
    # model_name = "Qwen/Qwen2.5-3B-Instruct-AWQ"  # 3B AWQ, ~2GB
    # model_name = "Qwen/Qwen2.5-7B-Instruct-AWQ"  # 7B AWQ, ~4GB

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Detect if model is pre-quantized (AWQ/GPTQ) by name
    is_quantized = "AWQ" in model_name or "GPTQ" in model_name

    if is_quantized:
        # Pre-quantized models need autoawq/auto-gptq (Linux only)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
        )
        print(f"Model loaded: {model_name} (pre-quantized)")
    elif is_splade:
        model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            dtype=torch.float16,
        )
        model = model.to(device)
        print(f"MLM Model loaded: {model_name} on {device}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
        )
        model = model.to(device)
        print(f"Model loaded: {model_name} on {device}")

    model.eval()

    return model, tokenizer