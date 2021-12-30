from parrot import Parrot
import torch
import warnings
from sumeval.metrics.rouge import RougeCalculator
from sumeval.metrics.bleu import BLEUCalculator
from nltk.translate.bleu_score import sentence_bleu
#import model_evaluation as Evals
#import nlp_metrics.bleu
warnings.filterwarnings("ignore")

''' 
uncomment to get reproducable paraphrase generations
def random_state(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

random_state(1234)
'''

#Init models (make sure you init ONLY once if you integrate this to your code)
parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")

#phrases = ["Can you recommed some upscale restaurants in Newyork?",
#           "What are the famous places we should not miss in Russia?"
#]
phrases = []
with open("samples.txt") as file:
  for line in file:
    phrases.append(line)

top_paraphrases = []
paraphrases = []
for phrase in phrases:
  print("-"*70)
  print("Input_phrase: ", phrase)
  print("-"*70)
  para_phrases = parrot.augment(input_phrase=phrase, use_gpu=False)
  next_paraphrases = []
  top_phrase = para_phrases[0]
  for para_phrase in para_phrases:
    print(para_phrase)
    if para_phrase[1] > top_phrase[1]:
      top_phrase = para_phrase
    next_paraphrases.append(para_phrase[0])
  paraphrases.append(next_paraphrases)
  top_paraphrases.append(top_phrase[0])
# evaluate paraphrases
print("/"*100)
summary_index = 0
for p in paraphrases:
  print("")
  original = phrases[summary_index]
  print("Generated Phrase: ", original)
  print(p)
  print("-"*70)
  bleu = BLEUCalculator()
  #score = bleu.bleu("I am waiting on the beach",
  #                  "He is walking on the beach")

  #score = bleu.bleu(original,new[0])
  score = sentence_bleu(p, original.split())
  print("BLEU: {}".format(score))
  # You need spaCy to calculate ROUGE-BE

  rouge = RougeCalculator(stopwords=True, lang="en")

  rouge_1 = rouge.rouge_n(
            summary=original,
            references=p,
            n=1)

  rouge_2 = rouge.rouge_n(
            summary=original,
            references=p,
            n=2)

  rouge_l = rouge.rouge_l(
            summary=original,
            references=p)

#  rouge_be = rouge.rouge_be(
#            summary="I went to the Mars from my living town.",
#            references=["I went to Mars", "It's my living town"])

  print("ROUGE-1: {}, ROUGE-2: {}, ROUGE-L: {}".format(rouge_1, rouge_2, rouge_l).replace(", ", "\n"))
  summary_index += 1
print("/"*100)
#for i in range(0,5):
#  original = phrases[i]
#  new = top_paraphrases[i]
#  print("")
#  print("Generated Phrase: ", type(original))
#  print("Paraphrase: ", top_paraphrases)
#  print("-"*70)

 