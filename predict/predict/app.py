from flask import Flask, request, render_template

from predict.predict.run import TextPredictionModel

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template(r'my-form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']

    model = TextPredictionModel.from_artefacts(r'C:\Users\zducreux\Desktop\EPF\From poc to prod\poc-to-prod-capstone\poc-to-prod-capstone\train\data\artefacts\2023-01-03-12-26-51')
    predictions = model.predict([text], 1)
    return predictions

if __name__ == "__main__":
    app.run(debug=True)