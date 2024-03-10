from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
import torch

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-33b-instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-33b-instruct", 
                                             trust_remote_code=True, 
                                             torch_dtype=torch.bfloat16,
                                             device_map="auto")


question = "Complete the method which accepts an array of integers, and returns one of the following:\n\n* `\"yes, ascending\"` - if the numbers in the array are sorted in an ascending order\n* `\"yes, descending\"` - if the numbers in the array are sorted in a descending order\n* `\"no\"` - otherwise\n\n\nYou can assume the array will always be valid, and there will always be one correct answer."
answer = '''
def is_sorted_and_how(arr):
    if arr == sorted(arr):
        return "yes, ascending"
    elif arr == sorted(arr, reverse=True):
        return "yes, descending"
    else:
        return "no"
'''

input_text = "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.\n\n"
input_text += "### Instruction: You are given a programming question and its corresponding answer in Python code. Carefully read both the question and answer. After, develop a step by step approach for solving the problem. Do not actually write any code.\n"
input_text += f"Question: {question}\n"
input_text += f"Answer: {answer}\n\n"
input_text += "### Response:\n"

print(input_text)

inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=512)
generation = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generation)