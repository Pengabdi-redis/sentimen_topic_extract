from flask import Flask, jsonify, request, render_template
from flask_restful import Resource, Api, reqparse
from datetime import datetime
import pickle
import ast
import json

app = Flask(__name__)
api = Api(app)

labels_date = []
values_ps = []
values_nets = []
values_negs = []

labels_words = []
values_count = []

with open('model.pkl','rb') as model_file:
    model = pickle.load(model_file)

class SentimentAnalysis(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument('kalimat', type=str, required=True, help='Kalimat yang akan dideteksi')
    def get(self):
        get_kalimat = SentimentAnalysis.parser.parse_args()
        sent = model.predict([get_kalimat['kalimat']])
        if sent[0] == -1:
            out = 'negatif'
        elif sent[0] == 0:
            out = 'netral'
        elif sent[0] == 1:
            out = 'positif'
        return {'sentimen':out}, 200

@app.route("/")
def chart():
    global labels_date, values_ps, values_nets, values_negs, labels_words, values_count
    labels_date = []
    values_ps = []
    values_nets = []
    values_negs = []
    labels_words = []
    values_count = []
    return render_template('main_chart.html', labels_date=labels_date, values_ps=values_ps, values_nets=values_nets, values_negs=values_negs, labels_words=labels_words, values_count=values_count)

@app.route('/refreshData')
def refresh_graph_data():
    global labels_date, values_ps, values_nets, values_negs, labels_words, values_count
    print("labels_date now: " + str(labels_date))
    print("values_ps now: " + str(values_ps))
    print("values_nets now: " + str(values_nets))
    print("values_negs now: " + str(values_negs))

    print("labels_words now: " + str(labels_words))
    print("values_count now: " + str(values_count))
    return jsonify(sLabels_date=labels_date, sValues_ps=values_ps, sValues_nets=values_nets, sValues_negs=values_negs, sLabels_words=labels_words, sValues_count=values_count)

@app.route('/updateData', methods=['POST'])
def update_data_post():
    global labels_date, values_ps, values_nets, values_negs, labels_words, values_count
    if not request.form:
        return "error",400
    if 'sentimen_neg_data' in request.form:
        values_negs_get = ast.literal_eval(request.form['sentimen_neg_data'])
        values_nets_get = ast.literal_eval(request.form['sentimen_net_data'])
        values_ps_get = ast.literal_eval(request.form['sentimen_pos_data'])
        labels_date_get = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        with open("data_sentimen.json","r") as myfile:
            data=ast.literal_eval(myfile.read())
        data["sentimen_neg_data"].append(values_negs_get)
        data["sentimen_net_data"].append(values_nets_get)
        data["sentimen_pos_data"].append(values_ps_get)
        data["labels_date"].append(labels_date_get)
        with open('data_sentimen.json', 'w') as json_file:
            json.dump(data, json_file)
        values_negs = data["sentimen_neg_data"]
        values_nets = data["sentimen_net_data"]
        values_ps = data["sentimen_pos_data"]
        labels_date = data["labels_date"]
        print("values negative received: " + str(values_negs))
        print("values netral received: " + str(values_nets))
        print("values positive received: " + str(values_ps))
        print("labels date: " + str(labels_date))
        return "success",201
    if 'labels_words_data' in request.form:
        labels_words = ast.literal_eval(request.form['labels_words_data'])
        values_count = ast.literal_eval(request.form['values_count_data'])
        print("labels words received: " + str(labels_words))
        print("values count received: " + str(values_count))
        return "success",201

if __name__ == "__main__":
    api.add_resource(SentimentAnalysis,'/sentimen_api/')
    app.run(host='localhost', port=5000)


