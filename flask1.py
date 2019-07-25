from flask import Flask, request, jsonify, render_template, make_response, abort, send_from_directory
from sklearn.externals import joblib

app = Flask(__name__)

# home route
@app.route('/')
def home():
    # return '<h1>Welcome!</h1>'
    return render_template('welcome.html')

# prediction page
@app.route('/prediksi')
def prediksi():
    return render_template('prediksi.html')

# result page
@app.route('/hasil', methods=['GET', 'POST'])
def hasil():
    if request.method == 'POST' :
        Aroma =float(request.form['Aroma'])
        Flavor = float(request.form['Flavor'])
        Aftertaste = float(request.form['Aftertaste'])
        Acidity = float(request.form['Acidity'])
        Sweetness = float(request.form['Sweetness'])
        Balance = float(request.form['Balance'])
        Mouthfeel = float(request.form['Mouthfeel'])
        if Aroma=="" or Flavor =="" or Aftertaste =="" or Acidity=="" or Sweetness=="" or Balance=="" or Mouthfeel=="" :
            return render_template('error.html')    

        prediksi = modelRF.predict([[
            Aroma, Flavor,Aftertaste, Acidity, Sweetness, Balance, Mouthfeel ]])[0]

        if int(prediksi) == 0:
            kesimpulan = 'Kopi yang cocok untuk anda adalah Arabica'
        else:
            kesimpulan = 'Kopi yang cocok untuk anda adalah Robusta'

        dataHasil={
            'Aroma':Aroma, 'Flavor':Flavor,'Aftertaste':Aftertaste, 'Acidity':Acidity, 'Sweetness':Sweetness, 'Balance':Balance, 'Mouthfeel':Mouthfeel,
            'PREDIKSI' : kesimpulan
        }
    return render_template('result.html', hasil=dataHasil)

@app.route('/images/<path:path>')   #root nya
def staticfile(path):
    return send_from_directory ('images', path)

@app.errorhandler(404)
def tidakFound(error):
    return make_response(
        render_template('error.html')
    )

@app.route('/NotFound')
def notFound():
    return render_template('error.html')


if __name__ == '__main__':
    modelRF = joblib.load('model1')
    app.run(debug = True, port = 1234)