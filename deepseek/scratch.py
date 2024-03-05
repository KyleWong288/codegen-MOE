from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
import torch

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()

input_text = "### Instruction: Write a Python program that solves the following question. Only output your code, and do not provide an explanation. \nQuestion: Complete the method which accepts an array of integers, and returns one of the following:\n\n* `\"yes, ascending\"` - if the numbers in the array are sorted in an ascending order\n* `\"yes, descending\"` - if the numbers in the array are sorted in a descending order\n* `\"no\"` - otherwise\n\n\nYou can assume the array will always be valid, and there will always be one correct answer. \n\n### Response:\n",
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=512)
generation = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generation)