import os
from flask import Flask, render_template
from flask import request

import pickle
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.python import keras

keras.backend.clear_session()

class IntentClassifier:
    def __init__(self,classes,model,tokenizer,label_encoder):
        self.classes = classes
        self.classifier = model
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder

    def get_intent(self,text):
        self.text = [text]
        self.test_keras = self.tokenizer.texts_to_sequences(self.text)
        self.test_keras_sequence = pad_sequences(self.test_keras, maxlen=16, padding='post')
        self.pred = self.classifier.predict(self.test_keras_sequence)
        return self.label_encoder.inverse_transform(np.argmax(self.pred,1))[0]

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
  
    if request.method == 'POST':
      form = request.form
    
      model = load_model('models/intents.h5')
        
      with open('utils/classes.pkl','rb') as file:
        classes = pickle.load(file)

      with open('utils/tokenizer.pkl','rb') as file:
        tokenizer = pickle.load(file)

      with open('utils/label_encoder.pkl','rb') as file:
        label_encoder = pickle.load(file)

      nlu = IntentClassifier(classes,model,tokenizer,label_encoder)  
        
      result = []
      sentence = form['sentence']
      prediction = nlu.get_intent(sentence)

      result.append(form['sentence'])
      result.append(prediction)

      return render_template("index.html",result = result)

    return render_template("index.html")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
