from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__,template_folder='html')
@app.route('/index', methods=['GET', 'POST'])
def home():
    return render_template('index.html')
@app.route('/csv_upload', methods=['GET', 'POST'])
def csv_upload():
    if request.method == 'POST':
        csv_file = request.files['csv_file']
        if not csv_file:
            return render_template('csv_upload.html', msg='No file selected')
        file_data = csv_file.read().decode('utf-8')
        lines = file_data.split("\n")
        csv_data = []
        for line in lines:
            csv_data.append( line.split(",") )
        df = pd.DataFrame(csv_data)
        return render_template('csv_upload.html', msg='CSV file successfully uploaded', df_view= df.to_html())

    return render_template('csv_upload.html')

if __name__ == '__main__':
    app.run(debug=True)