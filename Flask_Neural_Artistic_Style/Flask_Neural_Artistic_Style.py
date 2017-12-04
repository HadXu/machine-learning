import os
from flask import Flask, render_template, url_for, request, flash, redirect, send_from_directory, abort
from werkzeug.utils import secure_filename
from neural_network_model import model

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(app.static_folder, 'imgs')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'qwertyuioop'


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST' and 'photo' in request.files and 'optionsRadios' in request.form:
        file = request.files['photo']
        if file.filename == '':
            flash('no file name')
            return render_template('index.html')
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # 获取用户图片路径
        content_img = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # 获取样式图片路径
        option = request.form['optionsRadios']
        style_img = os.path.join(app.config['UPLOAD_FOLDER'], option)

        # 训练
        res = model.train(content_img, style_img)

        res.save(os.path.join(app.config['UPLOAD_FOLDER'], 'res.jpg'))

        return redirect(url_for('uploaded_file', filename='res.jpg'))
    return render_template('index.html')


@app.route('/')
def hello_world():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(port=8888, debug=True)
