import subprocess

# Your Python code stored in a string
code_string = """
# Your Python code here
name = input("Enter your name: ")
print("Hello, " + name + "!")
"""

# Run the code string as a subprocess
process = subprocess.Popen(['python', '-c', code_string], stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)

# Provide input to the subprocess
input_data = "John"  # Example input
stdout, stderr = process.communicate(input=input_data.encode())

# Decode and print the output
print(stdout.decode())
