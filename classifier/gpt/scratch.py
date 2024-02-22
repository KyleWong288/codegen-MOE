from dotenv import load_dotenv
from openai import OpenAI

if __name__ == "__main__":
    load_dotenv()
    client = OpenAI()

    premise = "You are given a programming question and a list of eight programming topics. Read the question, and choose one topic that you think is the most relevant for solving the question. Do not solve the question. Simply state your choice."
    skill_list_str = "[Amortized analysis, Bit manipulation, Complete search, Data structures, Dynamic programming, Greedy algorithms, Range queries, Sorting]"
    question = "Complete the method which accepts an array of integers, and returns one of the following:\n\n* `\"yes, ascending\"` - if the numbers in the array are sorted in an ascending order\n* `\"yes, descending\"` - if the numbers in the array are sorted in a descending order\n* `\"no\"` - otherwise\n\n\nYou can assume the array will always be valid, and there will always be one correct answer."
    prompt = f"{premise}\nList of topics: {skill_list_str}\nQuestion: {question}\n"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an intelligent expert at solving programming questions."},
            {"role": "user", "content": prompt} 
        ]
    )

    generation = response.choices[0].message.content
    print(generation)

