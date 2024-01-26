from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
# input_text = "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You may assume that each input would have exactly one solution, and you may not use the same element twice. You can return the answer in any order. #write an algorithm to solve this problem."
# inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
# outputs = model.generate(**inputs, max_length=1024)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

question1 = "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You may assume that each input would have exactly one solution, and you may not use the same element twice. You can return the answer in any order. #write an algorithm to solve this problem."
question2 = "#write an algorithm to solve the following problem. Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You may assume that each input would have exactly one solution, and you may not use the same element twice. You can return the answer in any order."
question3 = "You are given a 0-indexed 2D integer matrix grid of size n * n with values in the range [1, n * n]. Each integer appears exactly once except a which appears twice and b which is missing. The task is to find the repeating and missing numbers a and b. Return a 0-indexed integer array ans of size 2 where ans[0] equals to a and ans[1] equals to b. #write an algorithm to solve this problem."
question4 = "#write an algorithm to solve the following problem. You are given a 0-indexed 2D integer matrix grid of size n * n with values in the range [1, n * n]. Each integer appears exactly once except a which appears twice and b which is missing. The task is to find the repeating and missing numbers a and b. Return a 0-indexed integer array ans of size 2 where ans[0] equals to a and ans[1] equals to b."

questions = [question1, question2, question3, question4]
for i, question in enumerate(questions):
    inputs = tokenizer(question, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=1024)
    print("ANSWER:", i)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))