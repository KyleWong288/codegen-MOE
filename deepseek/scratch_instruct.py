from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()

question = '''
You are given a 0-indexed integer array nums of size 3 which can form the sides of a triangle.

A triangle is called equilateral if it has all sides of equal length.
A triangle is called isosceles if it has exactly two sides of equal length.
A triangle is called scalene if all its sides are of different lengths.
Return a string representing the type of triangle that can be formed or "none" if it cannot form a triangle.

Example 1:

Input: nums = [3,3,3]
Output: "equilateral"
Explanation: Since all the sides are of equal length, therefore, it will form an equilateral triangle.

Example 2:

Input: nums = [3,4,5]
Output: "scalene"
Explanation: As all the sides are of different lengths, it will form a scalene triangle.
'''

messages = [
    { "role": "user", "content": question}
]
# inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
# print("INPUT:")
# print(inputs)
# # 32021 is the id of <|EOT|> token
# outputs = model.generate(inputs, max_new_tokens=512, top_p=0.95, num_return_sequences=1, eos_token_id=32021)
# print("OUTPUT:")
# print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
print("INPUT:")
print(inputs)
# 32021 is the id of <|EOT|> token
outputs = model.generate(inputs, max_new_tokens=512, do_sample=True, top_p=0.95, num_return_sequences=5, eos_token_id=32021)
for i in range(len(outputs)):
    print("OUTPUT", i)
    print(tokenizer.decode(outputs[i][len(inputs[0]):]))
