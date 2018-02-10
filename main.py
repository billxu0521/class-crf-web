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
from flask import Flask
from flask import render_template
from config import DevConfig
from flask import request

# 初始化 Flask 類別成為 instance
app = Flask(__name__)
app.config.from_object(DevConfig)
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
    
    #while data:
    for index in range(len(data)):
        xseq, yref = data.pop(0)
        yout = tagger.tag(xseq)
        results.append(util.eval(yref, yout, "S"))
        lines.append(util.seq_to_line([x['gs0'] for x in xseq],yout,charstop))
    
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
    for line in lines:
        print (line.encode('utf8'))
        output = output + '<br>' + line
        print (line)
    output = score + '<br>' + output

    return (output)

if __name__ == '__main__':
    app.run()