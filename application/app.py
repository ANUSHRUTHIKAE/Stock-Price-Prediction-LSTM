
from flask import Flask, render_template , request
from test import stock_predict

app = Flask(__name__)

@app.get('/') 
def index(): 
    return render_template('home.html')

@app.route('/submit',methods=["POST"])
def submit():
    ticker = request.form['ticker']
    stock = request.form['stock']
    print(ticker,stock)
    stock_predict(ticker,stock)
    return render_template("index.html")


if __name__ == '__main__':   
    app.run(debug=True) 