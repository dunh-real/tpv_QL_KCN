# from mlx_lm import load, generate

# model, tokenizer = load("prism-ml/Bonsai-8B-mlx-1bit")

# prompt = "Write a story about Einstein"
# messages = [{"role": "user", "content": prompt}]
# prompt = tokenizer.apply_chat_template(
#     messages, add_generation_prompt = True
# )

# text = generate(model, tokenizer, prompt = prompt, verbose = True)
# print(text)

from transformers import pipeline

pipe = pipeline("image-to-text", model = "zai-org/GLM-OCR")

