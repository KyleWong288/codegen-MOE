code = '''
print("Hello, world!")
'''

try:
    compiled_code = compile(code, "<string>", "exec")
    exec(compiled_code)
except SyntaxError as e:
    print("Syntax error for compiled code:", e)
except Exception as e:
    print("Execution error for compiled code:", e)