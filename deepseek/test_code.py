import concurrent.futures
import json
import subprocess

NPROCS = 2

code1 = '''
n, m = map(int, input().split())
a = [None]*n
for i in range(n):
    s = input()
    cur = 0
    flag = 1
    for j in range(m):
        cur = cur * 26 + flag * (ord(s[j]))
        flag = -flag
    a[i]= [cur, i]
a.sort()
print(*[v[1] + 1 for v in a])
'''

code2 = '''
n,m=map(int,input().split())
l=[(lambda x:[i+1,[x[j]if j%2==0 else chr(155-ord(x[j]))for j in range(m)]])(input())for i in range(n)]
print(*[i[0]for i in sorted(l,key=lambda x:x[1])])
'''

generated_codes = [code1, code2]

# Args: code string, all inputs, all expected outputs, 
# For each test case, it creates a new process and passes in the input
def run_tests(args):

    code = args["code"]
    inputs = args["inputs"]
    expected_outputs = args["expected_outputs"]

    all_correct = 1
    for i in range(len(inputs)):
        input = inputs[i]
        expected_output = expected_outputs[i]
        expected_output2 = expected_output.replace(" \n", "\n")
        process = subprocess.Popen(['python', '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
        stdout, stderr = process.communicate(input=input.encode())
        output, error = stdout.decode(), stderr.decode()
        if error:
            print("ERROR:", error)
        if output != expected_output and output != expected_output2:
            all_correct = 0
            break
    return all_correct
            

# Evaluate all tests cases
# all_correct = 1
# for i in range(len(inputs)):
#     input = inputs[i]
#     expected_output = expected_outputs[i]
#     expected_output2 = expected_output.replace(" \n", "\n")
#     process = subprocess.Popen(['python', '-c', code_string], stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
#     stdout, stderr = process.communicate(input=input.encode())
#     output = stdout.decode()
#     if output != expected_output and output != expected_output2:
#         all_correct = 0
#         break


# Get input/output pairs
inputs = []
expected_outputs = []
with open("./cc_data/code_contests.jsonl", "r") as file:
    json_obj = json.loads(file.readline().strip())
    inputs = json_obj["generated_tests"]["input"]
    expected_outputs = json_obj["generated_tests"]["output"]



# # Provide input to the subprocess
# input = "5 2\nAA\nAB\nBB\nBA\nAZ\n"
# expected_output = "5 2 1 3 4 \n"
# expected_output2 = "5 2 1 3 4\n"
# stdout, stderr = process.communicate(input=input.encode())
# output = stdout.decode()

# # Decode and print the output
# print(output)
# print(output == expected_output)
# print(output == expected_output2)

# Set up args
args = [
    {
        "code": code1,
        "inputs": inputs,
        "expected_outputs": expected_outputs
    },
    {
        "code": code2,
        "inputs": inputs,
        "expected_outputs": expected_outputs
    },
]

with concurrent.futures.ThreadPoolExecutor(max_workers=NPROCS) as executor:
    # Submit the function to the executor for each argument
    future_to_argument = {executor.submit(run_tests, arg): arg for arg in args}
    # Retrieve results as they become available
    for future in concurrent.futures.as_completed(future_to_argument):
        arg = future_to_argument[future]
        try:
            result = future.result()
            # Process the result
            print("Received:", result)
        except Exception as e:
            # Handle exceptions
            print("Exception occurred for {}: {}".format(arg, e))
