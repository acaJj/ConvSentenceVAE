import numpy as np

import os
import config
import utils
import data
from train_ESIM import ESIM_pred, load_ESIM_model
import encoder_models

#%% 

from eval_metrics.bleu.bleu_scorer import BleuScorer
from eval_metrics.rouge.rouge import Rouge

# Define conventional automatic metrics warppers from eval_metrics package
def BLEU_score(target_sentence, pred_sentence, n_tokens=1):
    """Returns BLEU score at specified n-gram level for a given target and predicted sentence pair"""
    try:    
        # Set n to BLEU score level
        bleu_scorer = BleuScorer(n=n_tokens)
        bleu_scorer += (pred_sentence[0], target_sentence)
        BLEU_score, _ = bleu_scorer.compute_score()
        return np.around(BLEU_score[n_tokens-1], 4)
    except:
        print('rejected sentence: ', pred_sentence)

def ROUGE_score(target_sentence, pred_sentence):
    """Returns ROUGE score for a given target and predicted sentence pair"""
    try:
        rouge = Rouge()
        ROUGE_score = rouge.calc_score(pred_sentence, target_sentence)
        return np.around(ROUGE_score, 4)
    except:
        print('rejected sentence: ', pred_sentence)