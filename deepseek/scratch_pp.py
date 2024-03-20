import concurrent.futures

def run_tests(name):
    print("Hi", name)
    return 1

args = ["Alex", "Ben", "Carson", "Danny"]

num_procs = 4

with concurrent.futures.ThreadPoolExecutor(max_workers=num_procs) as executor:
    # Submit the function to the executor for each argument
    future_to_argument = {executor.submit(run_tests, arg): arg for arg in args}
    # Retrieve results as they become available
    for future in concurrent.futures.as_completed(future_to_argument):
        arg = future_to_argument[future]
        try:
            result = future.result()
            # Process the result
            print("Result for {}: {}".format(arg, result))
        except Exception as e:
            # Handle exceptions
            print("Exception occurred for {}: {}".format(arg, e))