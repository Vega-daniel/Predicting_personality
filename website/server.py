from flask import Flask,render_template, request,jsonify,Response
import pickle


#Create the app object that will route our calls
app = Flask(__name__)
# Add a single endpoint that we can use for testing

@app.route('/', methods = ['GET'])
def personality():
    return render_template('personality.html')

model = pickle.load(open('sgd.p','rb'))

@app.route('/inference',  methods = ['POST'])
def inference():
    req = request.get_json()
    text = req['text']
    prediction = model.predict([text])
    return jsonify({'prediction':prediction[0]})

if __name__ == '__main__':
	app.run(host='0.0.0.0',port = 4444,debug = True)
