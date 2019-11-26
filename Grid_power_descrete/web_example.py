from io import BytesIO

import pandas as pd
from flask import Flask, send_file

@app.route('/plot')
def plot():
    # your pandas code goes here ...

    plot = df.plot()
    stream = BytesIO()
    plot.figure.savefig(stream)
    stream.seek(0)
    return send_file(stream, mimetype='image/png')

app.run(debug=True, port=8000)