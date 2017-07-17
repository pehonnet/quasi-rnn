# -*- coding: utf-8-*-
from __future__ import print_function
from hyperparams import Hp
import codecs
import sugartensor as tf
import numpy as np
from prepro import *
from train import Graph
from nltk.translate.bleu_score import corpus_bleu
import sys


# if len(sys.argv) != 3 :
#     print("Usage : python translate.py sentences_to_translate output_file")
#     sys.exit(1)
sentences = sys.argv[1]
outputFile = sys.argv[2]

Hp.de_test = sentences

def eval(): 
    # Load graph
    g = Graph(mode="inference"); print("Graph Loaded")
        
    with tf.Session() as sess:
        # Initialize variables
        tf.sg_init(sess)

        # Restore parameters
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('asset/train'))
        print("Restored!")
        mname = open('asset/train/checkpoint', 'r').read().split('"')[1] # model name
        
        # Load data
        X, Sources = load_test_data_no_gt(input_reverse=Hp.reverse_inputs)
        char2idx, idx2char = load_vocab()
        
        with codecs.open(outputFile, "w", "utf-8") as fout:
            list_of_refs, hypotheses = [], []
            batchesToBeTranslated = (len(X) // Hp.batch_size ) * Hp.batch_size
            if batchesToBeTranslated != len(X):
                nMissing = len(X) - batchesToBeTranslated
                nToAdd = Hp.batch_size - nMissing
                # print("some incomplete batch, ", nMissing)
                # print(X)
                to_append = np.zeros((nToAdd, Hp.maxlen), np.int32) # * 0
                X = np.append(X,to_append, axis=0)
                for i in range(len(X) // Hp.batch_size):
                    # Get mini-batches
                    # print("minibatch ", i)
                    x = X[i*Hp.batch_size: (i+1)*Hp.batch_size] # mini-batch
                    sources = Sources[i*Hp.batch_size: (i+1)*Hp.batch_size]
                    # targets = Targets[i*Hp.batch_size: (i+1)*Hp.batch_size]
                
                    preds_prev = np.zeros((Hp.batch_size, Hp.maxlen), np.int32)
                    preds = np.zeros((Hp.batch_size, Hp.maxlen), np.int32)        
                    for j in range(Hp.maxlen):
                        # predict next character
                        outs = sess.run(g.preds, {g.x: x, g.y_src: preds_prev})
                        # update character sequence
                        if j < Hp.maxlen - 1:
                            preds_prev[:, j + 1] = outs[:, j]
                        preds[:, j] = outs[:, j]
                
                    # Write to file
                    for source, pred in zip(sources, preds): # sentence-wise
                        got = "".join(idx2char[idx] for idx in pred).split(u"␃")[0]
                        fout.write(source + " " + got + "\n")
                        # fout.write("- got: " + got + "\n\n")
                        fout.flush()

            else:
                print("using even number of files (minibatch size)")
                for i in range(len(X) // Hp.batch_size):
                    # Get mini-batches
                    # print("minibatch ", i)
                    x = X[i*Hp.batch_size: (i+1)*Hp.batch_size] # mini-batch
                    sources = Sources[i*Hp.batch_size: (i+1)*Hp.batch_size]
                    # targets = Targets[i*Hp.batch_size: (i+1)*Hp.batch_size]
                
                    preds_prev = np.zeros((Hp.batch_size, Hp.maxlen), np.int32)
                    preds = np.zeros((Hp.batch_size, Hp.maxlen), np.int32)        
                    for j in range(Hp.maxlen):
                        # predict next character
                        outs = sess.run(g.preds, {g.x: x, g.y_src: preds_prev})
                        # update character sequence
                        if j < Hp.maxlen - 1:
                            preds_prev[:, j + 1] = outs[:, j]
                        preds[:, j] = outs[:, j]
                
                    # Write to file
                    for source, pred in zip(sources, preds): # sentence-wise
                        got = "".join(idx2char[idx] for idx in pred).split(u"␃")[0]
                        fout.write(source + " " + got + "\n")
                        fout.flush()
                    
                        # For bleu score
			# Not applicable as reference does not exist
                        # ref = target.split()
                        # hypothesis = got.split()
                        # if len(ref) > 2:
                        #    list_of_refs.append([ref])
                        #    hypotheses.append(hypothesis)
            
                # Get bleu score
		# Not applicable as reference does not exist
                # score = corpus_bleu(list_of_refs, hypotheses)
                # fout.write("Bleu Score = " + str(100*score))
                                            
if __name__ == '__main__':
    eval()
    print("Done")
