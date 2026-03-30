from mlx_lm import load, generate

model, tokenizer = load("mlx-community/NVIDIA-Nemotron-3-Nano-4B-BF16")
prompt = "Hello, can you answer me in Vietnamese? What is the capital of Vietnam?"

if tokenizer._chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt = True, return_dict = False,
    )

response = generate(model, tokenizer, prompt = prompt, verbose = True)

print(response)