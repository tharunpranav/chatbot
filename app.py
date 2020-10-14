from flask import Flask,render_template,request
from flask_restful import Api
import json
import torch
from flask_restful import Resource
from model import NeuralNet
from main import tokenize,stem,bag_of_words
import random
import numpy
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/GET", methods=['GET'])
def get_bot_response():    
    userText = request.args.get('msg') 
    device = torch.device('cpu')
    k=os.getcwd()
    os.chdir(k)
    with open('data.json','r')as f:
        intents=json.load(f)
    FILE='data.pth'
    data=torch.load(FILE)
    input_size = data['input_size']
    hidden_size = data['hidden_size']
    output_size = data['output_size']
    all_words = data['all_words']
    tags = data['tags']
    model_state = data['model_state']      
    model = NeuralNet(input_size,hidden_size,output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()
    bot_name='BOT'
    
    while True:
        you=userText
        if you == 'qiut':
            break
        
        you = tokenize(you)
        X = bag_of_words(you, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)
        output=model(X)
        _,predicted  = torch.max(output , dim=1)
        tag=tags[predicted .item()]
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.70:
            print('--------------------------------------------------------------------')
            for intent in intents['intents']:
                if intent['tag'] == tag:
                    print('///////////////////////////////////////////////////////////////////////')
                    #print(f"{bot_name}:{random.choice(intent['responses'])}")
                    return f"{random.choice(intent['responses'])}"
        else:
            print('--------------------------------------------------------------------')
            #print(f'{bot_name}: i cont understand')
            return f'I cont understand :( but ill store this question in my database as soon as possible Tharun will train me with this questions'
          

if __name__ == "__main__":
    app.run()
