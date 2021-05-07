import flask
import numpy
import pickle
import pygad

# Use pickle to load in the pre-trained model.
fn = 'model/GA.pkl'

model_instance = pickle.load(open(fn, 'rb'))

app = flask.Flask(__name__, template_folder='pages')


@app.route('/', methods=['GET', 'POST'])


def main():
    if flask.request.method == 'GET':
        return flask.render_template('brrt.html') 
 
    if flask.request.method == 'POST':
        age = flask.request.form['age']
        anaemia = flask.request.form['anaemia']
        diabetes = flask.request.form['diabetes']
        high_blood_pressure = flask.request.form['high_blood_pressure']
        sex = flask.request.form['sex']
        smoking = flask.request.form['smoking']
        creatinine_phosphokinase = flask.request.form['creatinine_phosphokinase']
        ejection_fraction = flask.request.form['ejection_fraction']
        platelets = flask.request.form['platelets']
        serum_creatinine = flask.request.form['serum_creatinine']
        serum_sodium = flask.request.form['serum_sodium']
        
        input_variables = numpy.array([[age,anaemia,diabetes,high_blood_pressure,sex,smoking,creatinine_phosphokinase,
                                        ejection_fraction,platelets,serum_creatinine,serum_sodium]])

        array_inputs =  input_variables.astype(numpy.float)
        
        prediction = pygad.nn.predict(last_layer=model_instance, data_inputs=array_inputs)[0]
        
        pred = ''
        if int(prediction) == 0:
            pred = 'At Low Risk'
        elif int(prediction) == 1:
            pred = 'At High Risk'
        
        print(pred)
        
        return flask.render_template('brrt.html', result=str(pred))
    
if __name__ == '__main__':
    app.run()
