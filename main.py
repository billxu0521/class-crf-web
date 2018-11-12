#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 01:23:48 2018

@author: billxu
"""
import sys
import glob
import random
import pycrfsuite
import crf
import util
import datetime
import json
from urllib.parse import unquote
from flask import Flask, jsonify
from flask import render_template
from config import DevConfig
from flask import request
from flask import abort
from flask import render_template
from flask_cors import CORS

# 初始化 Flask 類別成為 instance
app = Flask(__name__)
CORS(app)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config.from_object(DevConfig)
tasks = [
    {
        'id': 1,
        'title': u'Buy groceries',
        'description': u'Milk, Cheese, Pizza, Fruit, Tylenol',
        'done': False
    },
    {
        'id': 2,
        'title': u'Learn Python',
        'description': u'Need to find a good Python tutorial on the web',
        'done': False
    }
]
# 路由和處理函式配對
@app.route('/')
def index():
    return render_template('index.html')
# 判斷自己執行非被當做引入的模組，因為 __name__ 這變數若被當做模組引入使用就不會是 __main__
if __name__ == '__main__':
    app.run()
    
@app.route('/crftag', methods=['POST'])
def predic():
    charstop = True # True means label attributes to previous char
    features = 3 # 1=discrete; 2=vectors; 3=both
    dictfile = 'vector/24scbow50.txt'
    modelname = 'datalunyu5001.m'
    vdict = util.readvec(dictfile)
    inputtext = request.form.get('input_text','')
    #li = [line for line in util.text_to_lines(inputtext)]
    li = util.text_to_lines(inputtext)
    
    print(li)
    data = []
    for line in li:
        x, y = util.line_toseq(line, charstop)
        print(x)
        if features == 1:
            d = crf.x_seq_to_features_discrete(x, charstop), y
        elif features == 2:
            d = crf.x_seq_to_features_vector(x, vdict, charstop), y
        elif features == 3:
            d = crf.x_seq_to_features_both(x, vdict, charstop), y
        data.append(d)
    
    tagger = pycrfsuite.Tagger()
    tagger.open(modelname)
    print ("Start testing...")
    results = []
    lines = []
    Spp = []
    Npp = []
    #while data:
    for index in range(len(data)):
        print(len(data))
        xseq, yref = data.pop(0)
        yout = tagger.tag(xseq)
        sp = 0
        np = 0
        for i in range(len(yout)):
            sp = tagger.marginal('S',i)
            Spp.append(sp) #S標記的機率
            print(sp)
            np = tagger.marginal('N',i) 
            Npp.append(np)#Nㄅ標記的機率
            print(np)
        results.append(util.eval(yref, yout, "S"))
        lines.append(util.seq_to_line([x['gs0'] for x in xseq],yout,charstop,Spp,Npp))
        #print(util.seq_to_line([x['gs0'] for x in xseq], (str(sp) +'/'+ str(np)),charstop))

    U_score = 0
    p_Scount = 0
    p_Ncount = 0
    for i in range(len(Spp)):
        _s = 0
        if Spp[i] > Npp[i]:
            _s = Spp[i]
        else :_s = Npp[i]
        _s = (_s - 0.5) * 10
        U_score = U_score + _s
        p_Scount = p_Scount + Spp[i]
        p_Ncount = p_Ncount + Npp[i]
   
    tp, fp, fn, tn = zip(*results)
    tp, fp, fn, tn = sum(tp), sum(fp), sum(fn), sum(tn)
    
    p, r = tp/(tp+fp), tp/(tp+fn)
    score = ''
    score = score + '<br>' + "Total tokens in Test Set:" + repr(tp+fp+fn+tn)
    score = score + '<br>' + "Total S in REF:" + repr(tp+fn)
    score = score + '<br>' + "Total S in OUT:" + repr(tp+fp)
    score = score + '<br>' + "Presicion:" + repr(p)
    score = score + '<br>' + "Recall:" + repr(r)
    score = score + '<br>' + "*******************F1-score:" + repr(2*p*r/(p+r))
    score = score + '<br>' + "======================="
    score = score + '<br>' + "character count:" + str(len(Spp))
    score = score + '<br>' +  "block uncertain rate:" + str((U_score / len(Spp)))

    
    output = ''
    key = 0
    for line in lines:
        #print (line.encode('utf8'))
        
        output = output + '<br>' + line 
        #print (line)
        key = key + 1
        
    #for index_m in ypp:
      #  output = output + '<br>' + line

    output = score + '<br>' + output 

    return (output)

#API用預測
def predic_api(inputtext):
    charstop = True # True means label attributes to previous char
    features = 3 # 1=discrete; 2=vectors; 3=both
    dictfile = 'vector/24scbow50.txt'
    modelname = 'datalunyu5001.m'
    vdict = util.readvec(dictfile)
    inputtext = inputtext
    #li = [line for line in util.text_to_lines(inputtext)]
    li = util.text_to_lines(inputtext)
    
    print(li)
    data = []
    for line in li:
        x, y = util.line_toseq(line, charstop)
        print(x)
        if features == 1:
            d = crf.x_seq_to_features_discrete(x, charstop), y
        elif features == 2:
            d = crf.x_seq_to_features_vector(x, vdict, charstop), y
        elif features == 3:
            d = crf.x_seq_to_features_both(x, vdict, charstop), y
        data.append(d)
    
    tagger = pycrfsuite.Tagger()
    tagger.open(modelname)
    print ("Start testing...")
    results = []
    lines = []
    Spp = []
    Npp = []
    out = []
    #while data:
    for index in range(len(data)):
        print(len(data))
        xseq, yref = data.pop(0)
        yout = tagger.tag(xseq)
        sp = 0
        np = 0
        for i in range(len(yout)):
            sp = tagger.marginal('S',i)
            Spp.append(sp) #S標記的機率
            print(sp)
            np = tagger.marginal('N',i) 
            Npp.append(np)#Nㄅ標記的機率
            print(np)
        results.append(util.eval(yref, yout, "S"))
        lines.append(util.seq_to_line([x['gs0'] for x in xseq],yout,charstop,Spp,Npp))
        #print(util.seq_to_line([x['gs0'] for x in xseq], (str(sp) +'/'+ str(np)),charstop))
        out.append(yout)
    tp, fp, fn, tn = zip(*results)
    tp, fp, fn, tn = sum(tp), sum(fp), sum(fn), sum(tn)
    
    p, r = tp/(tp+fp), tp/(tp+fn)
    score = ''
    score = score + '<br>' + "Total tokens in Test Set:" + repr(tp+fp+fn+tn)
    score = score + '<br>' + "Total S in REF:" + repr(tp+fn)
    score = score + '<br>' + "Total S in OUT:" + repr(tp+fp)
    score = score + '<br>' + "Presicion:" + repr(p)
    score = score + '<br>' + "Recall:" + repr(r)
    score = score + '<br>' + "*******************F1-score:" + repr(2*p*r/(p+r))
    
    output = ''
    print (lines)

    for line in lines:
        #line = unquote(line)
        print ("output:")
        print (line.encode('utf8'))
        #output = output + '<br>' + line
        output += line
        print (line)
    output = score + '<br>' + output

    #output = jsonify({'str': output})
    

    return (out)


@app.route('/signUp')
def signUp():
    return render_template('signUp.html')

@app.route('/signUpUser', methods=['POST'])
def signUpUser():
    #user =  request.form['username'];
    #password = request.form['password'];
    text = request.form['input_text']
    res = predic_api(text)
    return json.dumps({'status':'OK','data':res});

@app.route('/preseg', methods=['POST'])
def preseg():
    #user =  request.form['username'];
    #password = request.form['password'];
    text = request.form['input_text']    
    res = predic_api(text)
    return json.dumps({'status':'OK','data':res});

@app.route('/api/str/<string:inputtext>', methods=['GET'])
def get_task(inputtext):
    task = list(filter(lambda t: t['id'] == inputtext, tasks))
    #if len(task) == 0:
    #    abort(404)
    res = predic_api(inputtext)

    #return jsonify({'task': task[0]})
    return (res)

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)


