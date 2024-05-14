#!/usr/bin/python

from bottle import route, request, run, app, response
import bottle
import json
import os
import random
import string

# the decorator
def enable_cors(fn):
    def _enable_cors(*args, **kwargs):
        # set CORS headers
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'

        if bottle.request.method != 'OPTIONS':
            # actual request; reply with the actual response
            return fn(*args, **kwargs)

    return _enable_cors

def generate_random_name(length=12):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def create_folder_with_random_name():
    folder_name = generate_random_name()
    os.makedirs(os.path.join('pdf_content', folder_name), exist_ok=True)
    return folder_name

@route('/upload', method='POST')
@enable_cors
def receive_data():
    files = request.files.getall('pdf')

    if files:
        foldername = create_folder_with_random_name()
        for pdf_file in files:
            filename = pdf_file.filename
            save_path = os.path.join('pdf_content', foldername, filename)
            pdf_file.save(save_path)

        response_data = {
            'foldername': foldername,
            'message': f"PDFs guardados exitosamente en la carpeta {foldername}."
        }

        return json.dumps(response_data)
    else:
        return 400
    
if __name__ == '__main__':
    os.makedirs('pdf_content', exist_ok=True)
    run(host='localhost', port=8080, debug=True)   