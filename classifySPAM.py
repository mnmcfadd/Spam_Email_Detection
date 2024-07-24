'''
  This program shell reads email data for the spam classification problem.
  The input to the program is the path to the Email directory "corpus" and a limit number.
  The program reads the first limit number of ham emails and the first limit number of spam.
  It creates an "emaildocs" variable with a list of emails consisting of a pair
    with the list of tokenized words from the email and the label either spam or ham.
  It prints a few example emails.
  Your task is to generate features sets and train and test a classifier.

  Usage:  python classifySPAM.py  <corpus directory path> <limit number>
'''
# open python and nltk packages needed for processing
import os
import sys
import random
import nltk
from nltk.corpus import stopwords#######################################################
from nltk import FreqDist#######################################################
from nltk.collocations import *#######################################################
from nltk.metrics import ConfusionMatrix#######################################################

# define a feature definition function here
def find_features(document, spam_unigrams, ham_unigrams, spam_bigrams, ham_bigrams):#######################################################
  words = set(document)#######################################################
  document_bigrams = set(nltk.bigrams(document))#######################################################

  features = {}#######################################################
  # Handle spam unigrams
  for w in spam_unigrams:#######################################################
    features[f"spam_unigram {w}"] = (w in words)#######################################################
  # Handle ham unigrams
  for w in ham_unigrams:#######################################################
    features[f"ham_unigram {w}"] = (w in words)#######################################################
  # Handle spam bigrams
  for b in spam_bigrams:#######################################################
    features[f"spam_bigram {b}"] = (b in document_bigrams)#######################################################
  # Handle ham bigrams
  for b in ham_bigrams:#######################################################
    features[f"ham_bigram {b}"] = (b in document_bigrams)#######################################################

  return features#######################################################




# function to read spam and ham files, train and test a classifier 
def processspamham(dirPath,limitStr):
  # convert the limit argument from a string to an int
  limit = int(limitStr)
  
  # start lists for spam and ham email texts
  hamtexts = []
  spamtexts = []
  os.chdir(dirPath)
  # process all files in directory that end in .txt up to the limit
  #    assuming that the emails are sufficiently randomized
  for file in os.listdir("./spam"):
    if (file.endswith(".txt")) and (len(spamtexts) < limit):
      # open file for reading and read entire file into a string
      f = open("./spam/"+file, 'r', encoding="latin-1")
      spamtexts.append (f.read())
      f.close()
  for file in os.listdir("./ham"):
    if (file.endswith(".txt")) and (len(hamtexts) < limit):
      # open file for reading and read entire file into a string
      f = open("./ham/"+file, 'r', encoding="latin-1")
      hamtexts.append (f.read())
      f.close()
  
  # print number emails read
  print ("Number of spam files:",len(spamtexts))
  print ("Number of ham files:",len(hamtexts))
  print
  
  # create list of mixed spam and ham email documents as (list of words, label)
  emaildocs = []
  # add all the spam
  for spam in spamtexts:
    tokens = nltk.word_tokenize(spam)
    emaildocs.append((tokens, 'spam'))
  # add all the regular emails
  for ham in hamtexts:
    tokens = nltk.word_tokenize(ham)
    emaildocs.append((tokens, 'ham'))

  random.shuffle(emaildocs)
  
  
  # filtering + pre-processing tokens
  stopwords = nltk.corpus.stopwords.words('english') + ['subject','cc'] #######################################################

  for i, (tokens, label) in enumerate(emaildocs):#######################################################
    # converting alphabetical characters to lowercase for all tuples
    modified_tokens = [w.lower() for w in tokens]  #######################################################
    #updating emaildocs list with modified contents
    stopped_tokens =  [w for w in modified_tokens if not w in stopwords] #######################################################
    emaildocs[i] = (stopped_tokens, label) #######################################################

  # continue as usual to get all words and create word features
  
  #creating non-nested lists of all spam and ham tokens, respectively
  spamtokens = [tokens for (tokens, label) in emaildocs if label == 'spam']#######################################################
  hamtokens = [tokens for (tokens, label) in emaildocs if label == 'ham']#######################################################
  fullspamtokens = [token for sublist in spamtokens for token in sublist]#######################################################
  fullhamtokens = [token for sublist in hamtokens for token in sublist]#######################################################

  #using bag of words features/FreqDist to produce most frequently occurring tokens for each type of email
  spamdist = FreqDist(fullspamtokens)#######################################################
  hamdist = FreqDist(fullhamtokens)#######################################################
  spammctokens = spamdist.most_common(100)#######################################################
  hammctokens = hamdist.most_common(100)#######################################################
  lenspamtokens = len(fullspamtokens)#######################################################
  lenhamtokens = len(fullhamtokens)#######################################################
  print('Spam token word frequency')#######################################################
  for mcitem in spammctokens[:10]:#######################################################
    print(mcitem[0],'\t\t',mcitem[1]/lenspamtokens)#######################################################
  print('\nHam token word frequency')#######################################################
  for mcitem in hammctokens[:10]:#######################################################
    print(mcitem[0],'\t\t',mcitem[1]/lenhamtokens)#######################################################

  #Producing most frequently occurring bigrams for each type of email
  bigram_measures = nltk.collocations.BigramAssocMeasures()#######################################################

  spambfinder = BigramCollocationFinder.from_words(fullspamtokens)#######################################################
  hambfinder = BigramCollocationFinder.from_words(fullhamtokens)#######################################################
  spambscored = spambfinder.score_ngrams(bigram_measures.raw_freq)#######################################################
  hambscored = hambfinder.score_ngrams(bigram_measures.raw_freq)#######################################################

  print('\nSpam bigram frequency')#######################################################
  for spambbscore in spambscored[:10]:#######################################################
    print (spambbscore)#######################################################
  
  print('\nHam bigram frequency')#######################################################
  for hambbscore in hambscored[:10]:#######################################################
    print (hambbscore)#######################################################
  
  # Producing bigrams with highest mutual information scores
  spambfinder1 = BigramCollocationFinder.from_words(fullspamtokens)#######################################################
  hambfinder1 = BigramCollocationFinder.from_words(fullhamtokens)#######################################################
  spambfinder1.apply_freq_filter(20)#######################################################
  hambfinder1.apply_freq_filter(20)#######################################################
  spambscored1 = spambfinder1.score_ngrams(bigram_measures.pmi)#######################################################
  hambscored1 = hambfinder1.score_ngrams(bigram_measures.pmi)#######################################################

  print('\nSpam bigram by Mutual Information Score')#######################################################
  for spambbscore in spambscored1[:10]:#######################################################
    print (spambbscore)  #######################################################
  
  print('\nHam bigram by Mutual Information Score')#######################################################
  for hambbscore in hambscored1[:10]:#######################################################
    print (hambbscore)#######################################################
  
  # feature sets from a feature definition function
  spam_unigrams = [term for (term, freq) in spammctokens]#######################################################
  ham_unigrams = [term for (term, freq) in hammctokens]  #######################################################
  spam_bigrams = [term for (term, freq) in spambscored1[:100]]+ [term for (term, freq) in spambscored[:50]]#######################################################
  ham_bigrams = [term for (term, freq) in hambscored1[:100]] + [term for (term, freq) in hambscored[:50]]#######################################################
  
  # Run feature extraction function on all emaildocs
  feature_sets = [(find_features(email, spam_unigrams, ham_unigrams, spam_bigrams, ham_bigrams), label) for (email, label) in emaildocs]#######################################################

  #cross validation and obtaining measures of classifier performance
  num_folds = 5#######################################################
  subset_size = len(feature_sets) // num_folds#######################################################
  accuracy_scores = []#######################################################
  precision_scores = []#######################################################
  recall_scores = []#######################################################
  f_measure_scores = []#######################################################

  for i in range(num_folds):#######################################################
    testing_this_round = feature_sets[i*subset_size:][:subset_size]#######################################################
    training_this_round = feature_sets[:i*subset_size] + feature_sets[(i+1)*subset_size:]#######################################################

    classifier = nltk.NaiveBayesClassifier.train(training_this_round)#######################################################

    # Calculate Accuracy
    accuracy_scores.append(nltk.classify.accuracy(classifier, testing_this_round))#######################################################

    # Predictions and Gold Labels
    gold_labels = [label for (features, label) in testing_this_round]#######################################################
    predictions = [classifier.classify(features) for (features, label) in testing_this_round]#######################################################

    # Confusion Matrix
    cm = ConfusionMatrix(gold_labels, predictions)#######################################################

    # Calculate Precision, Recall, F-measure for 'spam' class
    TP = cm['spam', 'spam']  #######################################################
    FP = cm['ham', 'spam']   #######################################################
    FN = cm['spam', 'ham']   #######################################################

    if TP + FP:#######################################################
      precision = TP / (TP + FP)#######################################################
    else:#######################################################
      precision = 0#######################################################
    if TP + FN:#######################################################
      recall = TP / (TP + FN)#######################################################
    else:#######################################################
      recall = 0#######################################################
    if precision + recall:#######################################################
      f_measure = 2 * (precision * recall) / (precision + recall)#######################################################
    else:#######################################################
      f_measure = 0#######################################################

    precision_scores.append(precision)#######################################################
    recall_scores.append(recall)#######################################################
    f_measure_scores.append(f_measure)#######################################################

  # Calculate and Print Average Scores
  print("\nAverage Accuracy:", sum(accuracy_scores) / num_folds)#######################################################
  print("Average Precision:", sum(precision_scores) / num_folds)#######################################################
  print("Average Recall:", sum(recall_scores) / num_folds)#######################################################
  print("Average F-measure:", sum(f_measure_scores) / num_folds)#######################################################



"""
commandline interface takes a directory name with ham and spam subdirectories
   and a limit to the number of emails read each of ham and spam
It then processes the files and trains a spam detection classifier.

"""
if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print ('usage: python classifySPAM.py <corpus-dir> <limit>')
        sys.exit(0)
    processspamham(sys.argv[1], sys.argv[2])
        
