from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
from model.style_transfer import run_style_transfer
import PIL.Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'content_image' not in request.files or 'style_image' not in request.files:
        return 'No file part'
    content_file = request.files['content_image']
    style_file = request.files['style_image']
    if content_file.filename == '' or style_file.filename == '':
        return 'No selected file'
    if content_file and style_file:
        content_filename = secure_filename(content_file.filename)
        style_filename = secure_filename(style_file.filename)
        content_file.save(os.path.join(app.config['UPLOAD_FOLDER'], content_filename))
        style_file.save(os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
        result_image = run_style_transfer(os.path.join(app.config['UPLOAD_FOLDER'], content_filename),
                                          os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')
        result_image_pil = PIL.Image.fromarray(result_image)
        result_image_pil.save(result_path)
        return send_from_directory(app.config['UPLOAD_FOLDER'], 'result.jpg')

if __name__ == '__main__':
    app.run(debug=True)
