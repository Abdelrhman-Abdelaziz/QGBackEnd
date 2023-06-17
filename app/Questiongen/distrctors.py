from similarity.normalized_levenshtein import NormalizedLevenshtein
from sklearn.metrics.pairwise import cosine_similarity
import requests
import numpy as np
import better_profanity

normalized_levenshtein = NormalizedLevenshtein()

def filter_same_sense_words(wordlist):
  filtered_words=[]
  for eachword in wordlist:
    if eachword[0].split('|')[1]:
      filtered_words.append(eachword[0].split('|')[0].replace("_", " ").title().strip())
  return filtered_words

def get_highest_similarity_score(wordlist,wrd):
  score=[]
  for each in wordlist:
    score.append(normalized_levenshtein.similarity(each.lower(),wrd.lower()))
  return max(score)
def get_distractors_from_internet(word):
    url = "https://sense2vec.prod.demos.explosion.services/find"
    data = {
        "model": "2019",
        "sense": "auto",
        "word": word
    }

    response = requests.post(url, json=data)

    if response.status_code == 200:
        # Request successful
        result = response.json()
        # Process the result
        dis = [x["text"] for x in result["results"]]
        return dis
    else:
        # Request failed
        return None
def sense2vec_get_words_check(word,s2v):
    sense = s2v.get_best_sense(word)
    if sense is not None:
      return True
    else:
      return len(get_distractors_from_internet(word)) > 3
    
def sense2vec_get_words(word,s2v,topn,question):
    output = []
    output = get_distractors_from_internet(word)
    if(output == None):
      # print ("word ",word)
      try:
        sense = s2v.get_best_sense(word, senses= ["NOUN", "PERSON","PRODUCT","LOC","ORG","EVENT","NORP","WORK OF ART","FAC","GPE","NUM","FACILITY"])
        most_similar = s2v.most_similar(sense, n=topn)
      #   print (most_similar)
        output = filter_same_sense_words(most_similar)
        # print ("Similar ",output)
      except:
        output =[]

    threshold = 0.6
    final=[word]
    checklist =question.split()
    for x in output:
      if get_highest_similarity_score(final,x)<threshold and x not in final and x not in checklist:
        final.append(x)
    
    return final[1:]

def mmr(doc_embedding, word_embeddings, words, top_n, lambda_param):

    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphrase
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (lambda_param) * candidate_similarities - (1-lambda_param) * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]



def get_distractors (word,origsentence,sense2vecmodel,sentencemodel,top_n,lambdaval):
  distractors = sense2vec_get_words(word,sense2vecmodel,top_n,origsentence)
  # print ("distractors ",distractors)
  if len(distractors) ==0:
    return distractors
  # distractors_new = [word.capitalize()]
  # distractors_new.extend(distractors)
  # # print ("distractors_new .. ",distractors_new)

  # embedding_sentence = origsentence+ " "+word.capitalize()
  # # embedding_sentence = word
  # keyword_embedding = sentencemodel.encode([embedding_sentence])
  # distractor_embeddings = sentencemodel.encode(distractors_new)

  # # filtered_keywords = mmr(keyword_embedding, distractor_embeddings,distractors,4,0.7)
  # max_keywords = min(len(distractors_new),10)
  # filtered_keywords = mmr(keyword_embedding, distractor_embeddings,distractors_new,max_keywords,lambdaval)
  # # filtered_keywords = filtered_keywords[1:]
  filtered_keywords = distractors
  final = [word.capitalize()]
  for wrd in filtered_keywords:
    if wrd.lower() != word.lower() and word.lower() not in [string.lower() for string in wrd.split(" ")]:
      final.append(wrd.capitalize())
  
  # filter bad words
  final = [i for i in final if not(better_profanity.profanity.contains_profanity(i))]
  return final

def filter_keywords(keywords,s2v,fdist):
  retKeywords = [x for x in keywords if sense2vec_get_words_check(x,s2v) ]
  retKeywords = sorted(retKeywords, key=lambda x: fdist[x])
  return retKeywords

def filter_keywords2(keywords,s2v):
  retKeywords = [x for x in keywords if sense2vec_get_words_check(x,s2v) ]
  return retKeywords
