import numpy as np
import pickle
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
import model
from fs_gcsfs import GCSFS

# Load in data structures
with open("data/wordList.txt", "rb") as fp:
    wordList = pickle.load(fp)
wordList.append('<pad>')
wordList.append('<EOS>')

# Load in hyperparamters
vocabSize = len(wordList)
batchSize = 24
maxEncoderLength = 30
maxDecoderLength = maxEncoderLength
lstmUnits = 112
numLayersLSTM = 3

tf.reset_default_graph()

# Create placeholders
encoderInputs = [tf.placeholder(tf.int32, shape=(None,)) for i in range(maxEncoderLength)]
decoderLabels = [tf.placeholder(tf.int32, shape=(None,)) for i in range(maxDecoderLength)]
decoderInputs = [tf.placeholder(tf.int32, shape=(None,)) for i in range(maxDecoderLength)]
feedPrevious = tf.placeholder(tf.bool)

encoderLSTM = tf.nn.rnn_cell.BasicLSTMCell(lstmUnits, state_is_tuple=True)
#encoderLSTM = tf.nn.rnn_cell.MultiRNNCell([singleCell]*numLayersLSTM, state_is_tuple=True)
decoderOutputs, decoderFinalState = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(encoderInputs, decoderInputs, encoderLSTM, 
                                                            vocabSize, vocabSize, lstmUnits, feed_previous=feedPrevious)

decoderPrediction = tf.argmax(decoderOutputs, 2)

# Start session and get graph
sess = tf.Session()
#y, variables = model.getModel(encoderInputs, decoderLabels, decoderInputs, feedPrevious)

# Load in pretrained model
saver = tf.train.Saver()
# models = GCSFS(bucket_name='ml-chatbot-server-models')
# saver.restore(sess, tf.train.latest_checkpoint(models))
# 50K iteration model stored here: https://transfer.sh/qmmmA/pretrained_seq2seq.ckpt-50000.data-00000-of-00001
saver.restore(sess, tf.train.latest_checkpoint('models'))
zeroVector = np.zeros((1), dtype='int32')

def pred(inputString):
    inputVector = model.getTestInput(inputString, wordList, maxEncoderLength)
    feedDict = {encoderInputs[t]: inputVector[t] for t in range(maxEncoderLength)}
    feedDict.update({decoderLabels[t]: zeroVector for t in range(maxDecoderLength)})
    feedDict.update({decoderInputs[t]: zeroVector for t in range(maxDecoderLength)})
    feedDict.update({feedPrevious: True})
    ids = (sess.run(decoderPrediction, feed_dict=feedDict))
    return model.idsToSentence(ids, wordList)

# webapp
app = Flask(__name__, template_folder='./')

@app.route('/')
def main():
    return render_template('index.html')

@app.route("/get")
def get_bot_response():    
    userText = request.args.get('msg')    
    response =  pred(str(userText))
    return jsonify(response) 

if __name__ == '__main__':
    app.run(debug=True)