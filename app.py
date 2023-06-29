import os
import random
import string
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/<folder_name>', methods=['GET'])
def create_folder_and_read_file(folder_name):
    # Create folder
    folder_path = os.path.join(os.getcwd(), folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Generate random text
    random_text = ''.join(random.choices(string.ascii_letters + string.digits, k=10))

    # Create and write to the text file
    file_path = os.path.join(folder_path, 'random_text.txt')
    with open(file_path, 'w') as file:
        file.write(random_text)

    # Read the content of the text file
    with open(file_path, 'r') as file:
        file_content = file.read()

    return jsonify({'content': file_content})

if __name__ == '__main__':
    app.run()
