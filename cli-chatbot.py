from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "YOUR_MODEL_PATH_ALREADY_CONVERTED_TO_OPENVINO"
token_model = "YOUR_ORIGINAL_MODEL"
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer, pipeline
import time

tokenizer = AutoTokenizer.from_pretrained(token_model)
model = OVModelForCausalLM.from_pretrained(model_id, use_cache=True)
messages = [
    {"role": "system", "content": "You're are a helpful Assistant. Remember, maintain a natural tone. Be precise, concise, and casual"},
    {"role": "user", "content": "Tell me about favorite place in Sydney?"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to('cpu')

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
start_time_first_token = time.time()
with torch.no_grad():
    max_new_tokens=1
    outputs = model.generate(
    input_ids,
    max_new_tokens=max_new_tokens,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
first_token_latency = (time.time() - start_time_first_token) * 1000
first_response = outputs[0][input_ids.shape[-1]:]



start_time = time.time()
max_new_tokens=64
outputs = model.generate(
    input_ids,
    max_new_tokens=max_new_tokens,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
total_latency = (time.time() - start_time) * 1000
response = outputs[0][input_ids.shape[-1]:]
average_latency_per_token = total_latency / max_new_tokens
throughput = max_new_tokens / (total_latency / 1000)

print(f"First generated text : {tokenizer.decode(first_response, skip_special_tokens=True)}")
print(f"Total generated text : {tokenizer.decode(response, skip_special_tokens=True)}")
print(f"First token latency: {first_token_latency:.2f} ms")
print(f"Average latyency per token: {average_latency_per_token:.2f} ms")
print(f"Throughput: {throughput:.2f} tokens/s")
