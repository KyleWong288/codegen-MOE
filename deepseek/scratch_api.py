from dotenv import dotenv_values
from openai import OpenAI

API_KEY = dotenv_values(".env").get("DSC_API_KEY")

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com/v1")

question = "You are given a 0-indexed integer matrix grid and an integer k. Return the number of submatrices that contain the top-left element of the grid, and have a sum less than or equal to k. Example 1:\nInput: grid = [[7,6,3],[6,6,1]], k = 18\nOutput: 4\nExplanation: There are only 4 submatrices, shown in the image above, that contain the top-left element of grid, and have a sum less than or equal to 18."
answer = '''
def countSubmatrices(self, grid: List[List[int]], k: int) -> int:
    row = len(grid)
    col = len(grid[0])
    res = 0

    for i in range(row):
        for j in range(col):
            if i > 0:
                grid[i][j] += grid[i - 1][j]
            if j > 0:
                grid[i][j] += grid[i][j - 1]
            if i > 0 and j > 0:
                grid[i][j] -= grid[i - 1][j - 1]

            if grid[i][j] <= k:
                res += 1
            else:
                break

    return res
'''

input_text = "### Instruction: You are given a programming question and its corresponding answer in Python code. Carefully read both the question and answer. After, develop a step by step approach for solving the problem. Do not actually write any code.\n"
input_text += f"Question: {question}\n"
input_text += f"Answer: {answer}\n\n"
input_text += "### Response:\n"

print(input_text)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are an expert at analyzing data structures and algorithms"},
        {"role": "user", "content": input_text},
    ],
    max_tokens=1200
)

print(response.choices[0].message.content)