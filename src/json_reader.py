def read_first_lines(file_path, num_lines=1):
    """
    Reads the first few lines of a file and prints them.
    """
    try:
        with open(file_path, "r") as f:
            for i in range(num_lines):
                line = f.readline()
                if not line:
                    break
                print(line, end="")
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")


if __name__ == "__main__":
    # Assuming the JSON file is in the 'data' directory
    json_file_path = "../data/spy_options_data_19.json"
    read_first_lines(json_file_path)
