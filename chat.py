import json
import torch
from flask_restful import Resource
from model import NeuralNet
from main import tokenize,stem,bag_of_words
import random
import numpy
import os
class Bot(Resource):
    def get(self,you):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        os.chdir('C:/Users/user/Desktop/chat_bot/chatbot')
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
        print('Let chat!!!! type quit to exit')
        while True:
            you=you
            if you == 'qiut':
                break
            print(you)
            you = tokenize(you)
            X = bag_of_words(you, all_words)
            X = X.reshape(1, X.shape[0])
            X = torch.from_numpy(X).to(device)

            output=model(X)
            _,predicted  = torch.max(output , dim=1)
            tag=tags[predicted .item()]

            probs = torch.softmax(output, dim=1)
            prob = probs[0][predicted.item()]

            if prob.item() > 0.75:
                for intent in intents['intents']:
                    if intent['tag'] == tag:
                        #print(f"{bot_name}:{random.choice(intent['responses'])}")
                        print(random.choice(intent['responses']))
                        return f"{bot_name} : {random.choice(intent['responses'])}"

            else:
                #print(f'{bot_name}: i cont understand')
                return f'{bot_name}:I cont understand'
