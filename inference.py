from PIL import Image
import torch
import fire

from processing_paligemma import PaliGemmaProcessor
from modelling_gemma import KVCache, PaliGemmaForConditionalGeneration
from utils import load_hf_model


def move_inputs_to_device(model_inputs : dict, device : str):
    model_inputs = {k : v.to(device) for k, v in model_inputs.items()}
    return model_inputs


def _sample_top_p(probs : torch.Tensor, p : float):
    # (B, Vocab_Size)
    probs_sort, probs_idx = torch.sort(probs, dim = -1, descending = True)
    # (B, Vocab_Size)
    probs_sum = torch.cumsum(probs_sort, dim = -1)
    # (B, Vocab_Size)
    # Subtracting the "probs_sort" shifts the cumulative sum by 1 position to the right before masking
    mask = probs_sum - probs_sum > p
    # Zeroing out all the probabilities not selected by the Top P
    probs_sort[mask] = 0.0
    # Redistribute the probabilities to sum upto 1
    probs_sort.div(probs_sort.sum(dim = -1, keepdim = True))
    # Sample a token (index of the token) from the top p distribution
    next_token = torch.multinomial(probs_sort, num_samples = 1)
    # Get the token position in the vocabulary
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def get_model_inputs(
        processor : PaliGemmaProcessor, prompt : str, image_file_path : str, device : str
):
    image = Image.open(image_file_path)
    images = [image]
    prompts = [prompt]
    model_inputs = processor(text = prompts, images = images)
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs

def test_inference(
        model : PaliGemmaForConditionalGeneration,
        processor : PaliGemmaProcessor,
        device : str,
        prompt : str,
        image_file_path : str,
        max_tokens_to_generate : int,
        temperature : float,
        top_p : float,
        do_sample : bool,
):
    model_inputs = get_model_inputs(processor, prompt, image_file_path, device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"] 
    pixel_values = model_inputs["pixel_values"]

    kv_cache = KVCache()

    # Generating tokens until we see the stop token
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []

    for _ in range(max_tokens_to_generate):
        # Get the model outputs
        outputs = model(
            input_ids = input_ids, # Pre filling the KV Cache with the image tokens and placeholders
            pixel_values = pixel_values,
            attention_mask = attention_mask,
            kv_cache = kv_cache,
        )

        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]
        # Sampling the next token
        if do_sample:
            next_token_logits = torch.softmax(next_token_logits / temperature, dim = 1)
            next_token = _sample_top_p(next_token_logits, top_p)
        else:
            next_token = torch.argmax(next_token_logits, dim = -1, keepdim = True)
        assert next_token.size() == (1, 1)
        next_token = next_token.squeeze(0)
        generated_tokens.append(next_token)
        # Stop if the tokens have been generated
        if next_token.item() == stop_token:
            break

def main(
        model_path : str = None,
        prompt : str = None,
        image_file_path : str = None,
        max_tokens_to_generate : int = 100,
        temperature : float = 0.8,
        top_p : float = 0.9,
        do_smaple : bool = False,
        only_cpu : bool = False,
):
    device = "cpu"
    
    if not only_cpu:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
    
    print("Device in use :", device)
    print(f"Loading Model")
    model, tokenizer = load_hf_model(model_path, device)
    model = model.to(device).eval()

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size

    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    print("Running Inference")
    with torch.no_grad():
        test_inference(
            model,
            processor,
            device,
            prompt,
            image_file_path,
            max_tokens_to_generate,
            temperature,
            top_p,
            do_smaple
        )


if __name__ == "main":
    fire.Fire(main)