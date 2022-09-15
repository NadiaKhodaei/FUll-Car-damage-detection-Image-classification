import os
import json
import urllib
import h5py
import pickle as pk
import numpy as np

from os.path import join, dirname, realpath
from flask import Flask, request, redirect, url_for, send_from_directory, render_template, flash, Response
from werkzeug.utils import secure_filename

import engine 
# A <form> tag is marked with enctype=multipart/form-data and an <input type=file> is placed in that form.
# The application accesses the file from the files dictionary on the request object.
# use the save() method of the file to save the file permanently somewhere on the filesystem.

UPLOAD_FOLDER = join(dirname(realpath(__file__)), 'static/uploads/') # where uploaded files are stored
ALLOWED_EXTENSIONS = set(['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG', 'gif', 'GIF']) # models support png and gif as well

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 # max upload - 10MB
app.secret_key = 'secret'

# check if an extension is valid and that uploads the file and redirects the user to the URL for the uploaded file
def allowed_file(filename):
	return '.' in filename and \
		   filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def home():
	return render_template('index.html', result=None)

@app.route('/assessment')
def assess():
	return render_template('index.html', result=None)

@app.route('/assessment', methods=['GET', 'POST'])
def upload_and_classify():
	if request.method == 'POST':
		if 'file' not in request.files:
			flash('No file part')
			return redirect(url_for('assess'))

		file = request.files['file']
		if file.filename == '':
			flash('No selected file')
			return redirect(url_for('assess'))

		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename) # used to secure a filename before storing it directly on the filesystem
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
			model_results = engine.engine(filepath)

			return render_template('results.html', result=model_results, filename=filename)
	
	flash('Invalid file format - please try your upload again.')
	return redirect(url_for('assess'))

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
	return send_from_directory(app.config['UPLOAD_FOLDER'],
							   filename)

if __name__ == '__main__':
	app.run(host='0.0.0.0',port=8080,debug=True, use_reloader=False) 	
