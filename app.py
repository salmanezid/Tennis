from flask import Flask,render_template,url_for,request,redirect,session,send_file
from datetime import datetime

from io import BytesIO
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/videos'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/',methods=['POST','GET'])
def dashboard():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part', 400

        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400
        try:
            # Upload the file directly from memory to S3
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(video_path)
            video_url = url_for('static', filename=f'videos/{file.filename}')
            return render_template('output.html',video_url=video_url,video_url2=video_url)
        except Exception as e:
            return f"An error occurred: {e}", 500

    return render_template('upload.html')
@app.route('/output',methods=['POST','GET'])
def output():

    return render_template('output.html')


if __name__=="__main__":
    app.run(debug=True,host='0.0.0.0')