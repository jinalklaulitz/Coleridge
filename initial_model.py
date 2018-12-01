#!/opt/conda/envs/nlpEnv/bin python3

######################################
#RCC-04 Model "Halal"
#Created by: Urwa Muaz & Tanya Nabila
#Maintained by: Marvin Mananghaya
#######################################
# Required folder structure
#
#                 |- files (txt,pdf)   
# | -- train_test |
# |              
# |
# | -- main folder <where this script is>
#
#Datasources:
# folder path: ../train_test
#   1.) publications.json
#   2.) data_sets.json
#   3.) data_set_citations.json
#   4.) sage_research_fields.json  
#
# folder path: . (main folder)
#   1.) CNNtokenizer.pickle
#   2.) CNNmodel.json 
#   3.) CNNmodel.h5
#   4.) dataset_vocab_production.txt
#   5.) datasets_lines_production.txt
#   6.) labelledTextFiles.txt 
#   7.) sage_fields_vocab.txt
#   8.) fields_lines.txt
#   9.) abbbreviations.json	
#   10.) bmvocab.txt
#
#Output:
# folder path: 
#   1.) data_set_mentions.json 
#   2.) data_set_citations.json 
#
########################################################################Declare/Import Libraries##################################################3
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
import string
import json
import numpy as np
import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
import string
import json
import numpy as np
import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import spacy
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout
from keras import regularizers
from keras.models import model_from_json
import pickle
#Data Processing related declaration
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

#####Declare Parameters or initialize objects#####

#Load large english model for spacy
nlp = spacy.load('en_core_web_lg')

#create PorterStemmer instance
ps = PorterStemmer()

#params
TEXT_DIRECTORY = '../train_test/files/text/'
CNN_TOKENIZER_File = 'CNNtokenizer.pickle'
CNN_MODEL_FILE = 'CNNmodel.json'
CNN_MODEL_WEIGHTS_FILE = "CNNmodel.h5"
DATASET_VOCAB_FILE = 'dataset_vocab_production.txt'
CNN_VOCAB_FILE = 'bmvocab.txt'
DATSETS_JSON_FILE = '../train_test/data_sets.json'
PROCESSED_DATASET_LINES = 'datasets_lines_production.txt'
PUBLICATIONS_JSON_FILE = '../train_test/publications.json'
OUTPUT_DIRECTORY = './output/'
SAGE_FIELDS_FILE = '../train_test/sage_research_fields.json'
SAGE_VOCAB_FILE = 'sage_fields_vocab.txt'
SAGE_FIELDS_LINES = 'fields_lines.txt'

#############################################################################Declare Helper Functions################################################

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# saves list
def save_list(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

#NER based sentence filter methods
def containsEntity(entities, sentence):
    for e in entities:
        if e.start >= sentence.start and e.end <= sentence.end:
            return True
    return False

def excludeReference(text):
    tokens = text.split()
    l = len(tokens)
    if 'REFERENCES' in tokens:
        ind = l-1-tokens[::-1].index('REFERENCES')
    elif 'References' in tokens:
        ind = l-1-tokens[::-1].index('References')
    else:
        ind = l
    return ' '.join(tokens[:ind])

##load abbreviations
## Marvin note to self: work on refactoring this in the future
##  abbreviations.json at pwd
file = 'abbreviations.json'
abbtext = load_doc(file)
abbreviations = json.loads(abbtext)

def findAbbreviation(sentence):
    regex = r"\b[A-Z][A-Z]+\b"
    abbreviations = re.findall(regex, sentence)
    return abbreviations

def expandAbbreviation(sentence, abbdict):
    abbs = findAbbreviation(sentence)
    for a in abbs:
        if a in abbdict:
            sentence = sentence.replace(a,abbdict[a][0])
    return sentence

def specialMapping(word):
    if word == 'studi':
        return 'survey'
    else:
        return word

# turn a doc into clean tokens
def clean_doc(doc):
    # abbreviation disambiguation
    doc = expandAbbreviation(doc, abbreviations)
    # split into tokens by white space
    tokens = doc.split()
    # Exclude text below references
    #tokens = excludeReference(tokens)
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # make lower case
    tokens = [word.lower() for word in tokens]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # stemming
    tokens = [ps.stem(word) for word in tokens]
    #specialMapping
    tokens = [specialMapping(word) for word in tokens]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

# load doc, clean and return line of tokens
def doc_to_line(sentence, vocab):
   # clean doc
    tokens = clean_doc(sentence)
   # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)

# load all docs in a directory
def process_docs(sentences, vocab):
    lines = list()
    # walk through all files in the folder
    for sentence in sentences:
        # load and clean the doc
        line = doc_to_line(sentence, vocab)
        # add to list
        lines.append(line)
    return lines

def load_Keras_Tokenizer_CNN(pickleFilePath):
    with open(pickleFilePath, 'rb') as handle:
       tokenizer = pickle.load(handle)
    return tokenizer

def load_CNN_Sentece_Classifier(modelFile, weightsFile):
    # load json and create model
    json_file = open(modelFile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)   
    # load weights into new model
    model.load_weights("CNNmodel.h5")
    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Loaded model from disk")
    return model

def loadVocab(vocabFile):
    vocab = load_doc(vocabFile)
    vocab = vocab.split()
    vocab = set(vocab)
    return vocab

def loadDataSetTitlesAndIds(datasetJsonFile):
    text = load_doc(datasetJsonFile)
    loaded_json_all = json.loads(text)
              
    dataSetIds = [ dataset['data_set_id'] for dataset in loaded_json_all ]
    dataSetTitles = [ dataset['title'] for dataset in loaded_json_all ]
    
    return dataSetIds,dataSetTitles

def getLabelLength(labels):
    return [len(l) for l in labels]

def treatDates(tokens):
    rawDates = []
    for token in tokens:
        if len(token) == 6: #200108
            if re.match(r'([1-2][0,9][0-9]{4})', token):
                start = int(token[0:4])
                end = int(token[:2]+token[4:6])
                if (end>start):
                    years = list( range(start,end+1) )
                    years = [str(y) for y in years]
                    rawDates.append(token)
                    tokens += years
        if len(token) == 8: #20012008
            if re.match(r'([0-2][0,9][0-9]{2}[0-2][0,9][0-9]{2})', token):
                start = int(token[0:4])
                end = int(token[4:8])
                if (end>start):
                    years = list( range(start,end+1) )
                    years = [str(y) for y in years]
                    rawDates.append(token)
                    tokens += years
    tokens = [t for t in tokens if t not in rawDates]
    return tokens

def clean_mention(doc):
    # abbreviation disambiguation
    doc = expandAbbreviation(doc, abbreviations)
    
    # split into tokens by white space
    tokens = doc.split()

    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    
    # make lower case
    tokens = [word.lower() for word in tokens]
    
    # remove remaining tokens that are not alphabetic
    #tokens = [word for word in tokens if not word.isalpha()]
    tokenTemp = tokens.copy()
    # treat dates
    tokens = treatDates(tokens)
    
    # filter out stop words
    #stop_words = set(stopwords.words('english'))
    #tokens = [w for w in tokens if not w in stop_words]
    
    # stemming
    tokens = [ps.stem(word) for word in tokens]
    
    #specialMapping
    tokens = [specialMapping(word) for word in tokens]
    

    return " ".join(tokens)

def min_max_scale(X):
    mini = np.min(X)
    #mini = 0
    maxi = np.max(X)
    if maxi == 0:
        return X
    return (X - mini) / (maxi - mini)

def min_max_2D(X):
    rowList = []
    for i in range(X.shape[0]):
        rowList.append(min_max_scale(X[i]))
    return np.array(rowList)

def zscore_scale(X):
    #mini = np.min(X)
    mean = np.mean(X)
    std = np.std(X)
    if std == 0:
        return np.zeros(len(X))
    return (X - mean) / (std)

def zscore_scale_2D(X):
    rowList = []
    for i in range(X.shape[0]):
        rowList.append(min_max_scale(X[i]))
    return np.array(rowList)

#debuf functions

def print_Hits(sentence,score):
    s,d = zip(*sorted(zip(score, sentence)))
    for i in range(len(s)):
        print(s[i])
        print(d[i])
        print()

#Marvin: need to refactor this
def getTrueLabels(samplePublications):
    sampleTextFiles = [p['text_file_name'] for p in samplePublications]
    file = 'publications.json' ##
    directory = '../train_test/' #go back to me marvin#
    pub_json = pd.read_json(directory+file)

    pub_json.head()

    pub_json = pub_json[['text_file_name','publication_id']]
    pub_json = pub_json[pub_json.text_file_name.apply(lambda x: x in sampleTextFiles)]
    pub_json.head()

    file = 'data_set_citations.json'
    directory = '../train_test/' #go back to me marvin#
    cit = pd.read_json(directory+file)

    cit = cit[['data_set_id', 'publication_id']]
    cit.head()

    pub_json = pub_json.merge(cit, left_on=['publication_id'], right_on=['publication_id'], how='left')
    pub_json.head()

    len(pub_json)

    pub_json = pub_json[['text_file_name','data_set_id']]
    pub_json.columns =  ['file','data_set_id']
    pub_json['file'] = pub_json.file.apply(lambda x : x.split('.')[0])
    return pub_json

def evaluationMetrics(trueLabels,resultsDF):
    truePair = set(zip(trueLabels.file.values, trueLabels.data_set_id.values))
    predPair = set(zip(resultsDF.pubId.values, resultsDF.did.values))
    recall = len(truePair.intersection(predPair)) / len(truePair)
    precision = len(truePair.intersection(predPair)) / len(predPair)
    fscore = 2 * (precision * recall) / (precision + recall)
    return recall,precision,fscore

def save_citations(DF):
    data = DF.to_dict('records')
    with open(OUTPUT_DIRECTORY+'data_set_citations.json', 'w') as fp:
        json.dump(data, fp)

def save_mentions(DF):
    data = DF.to_dict('records')
    with open(OUTPUT_DIRECTORY+'data_set_mentions.json', 'w') as fp:
        json.dump(data, fp)

#fix  (marvin:???)
def getFileNames():
    txtFiles = os.listdir(TEXT_DIRECTORY)
    txtFiles = [t for t in txtFiles if not (t.startswith(".") and t.endswith('.txt')) ]
    labelledFiles = load_doc('labelledTextFiles.txt').split('\n')
    unlabbellledtxtFiles = [t for t in txtFiles if t not in labelledFiles]
    print("FileCount:",len(txtFiles))
    return labelledFiles

## Returns ORG entities and sentences
def parseSpacy(text):
    text = excludeReference(text)
    spacydoc = nlp(text)
    sentences = list(spacydoc.sents)
    entities = [e for e in spacydoc.ents if e.label_ == 'ORG']
    print("Sentences: ",len(sentences))
    return sentences,entities

def filterSentencesByNer(sentences,entities):
    filteredSentences = [s.text for s in sentences if containsEntity(entities, s)]
    filteredSentences = list(set(filteredSentences))
    print("Filtered Sentences: ",len(filteredSentences))
    return filteredSentences

def removeSpecialCharacters(sentences):
    sentences = [s.replace('\n',' ') for s in sentences]
    sentences = [s.replace('\xad', '') for s in sentences]
    return sentences

def getMentionSentencesDF(filteredSentences, y_hat, y_prob, threshHold):
    DF = pd.DataFrame({'sentence':filteredSentences,'Pscore': y_prob})
    DF = DF[DF.Pscore > threshHold]
    print('Dataset Mentions: ',len(DF))
    return DF

def getDataSetProcessedLines():
    # each field as a sentence
    docs = load_doc(PROCESSED_DATASET_LINES).split('\n')
    return docs

def getDatasetNgramVectorizer(docs):
    # create the tokenizer
    vectorizer = TfidfVectorizer(ngram_range=(2, 4))
    # fit the tokenizer on the documents
    tfidVec = vectorizer.fit(docs)
    return tfidVec

def getSimilarityMatrix_sent_datasets(hitSents,vectorizerDataset,dataset_Ngram):   
    # prepare negative reviews
    sents = process_docs(hitSents, dataset_vocab)    
    # encode training data set
    sents_Ngram = vectorizerDataset.transform(sents)
    #print('Sentence TFID shape: ',sents_Ngram.shape)
    Cos_Sim = cosine_similarity(sents_Ngram, dataset_Ngram, dense_output=True)
    print('Cos Sim shape: ',Cos_Sim.shape)
    return Cos_Sim

def getDatasetCandidateMatchesDF(df,Cos_Sim,dataSetIds,dataSetTitles, sim_threshHold,pubId):  
    #Cos_Sim = min_max_2D(Cos_Sim)
    DataLabel = []
    DataTitle = []
    sim_score = []
    for i in range(Cos_Sim.shape[0]):
        did = []
        dtit = []
        sscr = []
        for j in range(len(Cos_Sim[0])):
            if(Cos_Sim[i][j] > sim_threshHold):
                did.append(dataSetIds[j])
                dtit.append(dataSetTitles[j])
                sscr.append(Cos_Sim[i][j])
        DataLabel.append(did)
        DataTitle.append(dtit)
        sim_score.append(sscr)
    
    df['matches'] = getLabelLength(DataLabel)
    df['datasetIds'] = DataLabel
    df['data_sim_scores'] = sim_score
    df['datasetTitles'] = DataTitle
    df['pubID'] = pubId
    return df

def mergeSimilarDatasets(DF):
    datasetgroups = []
    for i in range(len(DF.datasetTitles)):
        tit = DF.datasetTitles.values[i]
        if not tit:
            continue
        found = False
        for j in range(len(datasetgroups)):
            if len(set(datasetgroups[j]).intersection(set(tit))) > 0:
                datasetgroups[j] = list(set(datasetgroups[j]).union(set(tit)))
                found = True
                break
        if not found:
            datasetgroups.append(tit)
    print('Numer of Groups: ',len(datasetgroups))
    print('Group Sizes',[len(d) for d in datasetgroups])
    return datasetgroups

def getGroupHitsSimMatrix(currentgroup,groupHits):
    # each field as a sentence
    docs = [ clean_mention(c) for c in currentgroup ]
    # create the tokenizer
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    # fit the tokenizer on the documents
    bow = vectorizer.fit(docs)
    # encode training data set
    sentence_bow = bow.transform(docs)
    #print(sentence_bow.shape)
    #each field as a sentence
    docs = [" ".join([ clean_mention(c) for c in groupHits])] 

    # encode training data set
    doc_bow = bow.transform(docs)
    #print(doc_bow.shape)
    
    data_sim = cosine_similarity(doc_bow, sentence_bow, dense_output=True)
    
    group_sim_scores = min_max_scale(data_sim.reshape(data_sim.shape[1]))
    return group_sim_scores

def citationScoring(mentions,score):
    scrs = []
    for i in range(len(score)):
        if len(mentions[i]) == 1:
            print(1)
            scrs.append(0.6 * score[i])
        if len(mentions[i]) == 2:
            print(2)
            scrs.append(0.7 * score[i])
        if len(mentions[i]) == 3:
            scrs.append(0.8 * score[i])
        if len(mentions[i]) >= 4:
            scrs.append(1 * score[i])       
    return scrs

def generateResults(datasetgroups,dataset_name_to_Id,DF,pubId,GROUP_SIM_THRESHHOLD):
    
    finalLabelList = []
    for i in range(len(datasetgroups)):
        currentgroup = datasetgroups[i]
        
        groupHits = [s for s,t in zip(DF.sentence,DF.datasetTitles) if len(set(t).intersection(set(currentgroup))) > 0 ]  #  
        
        if(len(currentgroup)<2):
            row = {}
            row['publication_id'] = pubId
            row['data_set_id'] = dataset_name_to_Id[currentgroup[0]]
            row['score'] = 1
            row['mention_list'] = groupHits
            finalLabelList.append(row)
            continue
        
        group_sim_scores = getGroupHitsSimMatrix(currentgroup,groupHits)
        print('Group ',i,' : ',group_sim_scores.shape)
        
        #print_Hits(currentgroup, group_sim_scores)
        hit_tit_scr = [ (tit,scr) for tit,scr in zip(currentgroup,group_sim_scores) if scr > GROUP_SIM_THRESHHOLD ]
        
    
        for hid in hit_tit_scr:
            row = {}
            row['publication_id'] = pubId
            row['data_set_id'] = dataset_name_to_Id[hid[0]]
            row['score'] = hid[1]
            row['mention_list'] = groupHits
            finalLabelList.append(row)
            
    finalLabelDF = pd.DataFrame(finalLabelList)
    if len(finalLabelDF) > 0:
        finalLabelDF['score'] = citationScoring(finalLabelDF.mention_list.values, finalLabelDF.score.values)
    return finalLabelDF

def mentionScoring(prob,matches,avgSimScore):
    if matches > 4:
        return 1
    if matches == 0:
        return 0.7 * prob
    if avgSimScore >= 0.5:
        return 0.9 * prob
    if avgSimScore < 0.5:
        return 0.8 * prob

def getMentionsResults(DF):
    DF['AvgSimScore'] = DF.data_sim_scores.apply(lambda x: np.mean(x) if len(x)>0 else 0)
    calScore = np.vectorize(mentionScoring)
    score = calScore(DF.Pscore.values, DF.matches.values, DF.AvgSimScore.values)
    DF['score'] = score
    DF = DF[['pubID','sentence','score']]
    DF.columns = ['publication_id', 'mention', 'score']
    return DF


##main function
def runPipeLine(publications, max_seq_len, hit_th , sim_th, group_sim_th):
    citationsDF = None
    mentionsDF = None
    for pub in publications:
        file = pub['text_file_name']
        pubId = pub['publication_id']
        txt = load_doc(TEXT_DIRECTORY+file)
        sentences,entities = parseSpacy(txt)
        filteredSentences = filterSentencesByNer(sentences,entities)
        filteredSentences = removeSpecialCharacters(filteredSentences)
        
        processed_lines = process_docs(filteredSentences, cnnVocab)
        encoded_docs = cnnTokenizer.texts_to_sequences(processed_lines)
        processed_sequences = pad_sequences(encoded_docs, maxlen=max_seq_len, padding='post')
        y_prob = model.predict(processed_sequences).reshape(len(processed_lines))
        y_hat = model.predict_classes(processed_sequences).reshape(len(processed_lines))

        classifierResultDF = getMentionSentencesDF(filteredSentences, y_hat, y_prob, hit_th)

        if len(classifierResultDF) < 1:
            print("No mentions for file : ",file)
            continue
    
        cosineSim_sent_dataset = getSimilarityMatrix_sent_datasets(classifierResultDF.sentence.values, \
                                                          vectorizerDataset,dataset_Ngram)

        candidateMatchesDF = getDatasetCandidateMatchesDF(classifierResultDF,cosineSim_sent_dataset, \
                                                          dataSetIds,dataSetTitles,sim_th,pubId)
        
        datasetGroupsTitles = mergeSimilarDatasets(candidateMatchesDF)

        resDf = generateResults(datasetGroupsTitles,dataset_name_to_Id,candidateMatchesDF,pubId, \
                                group_sim_th)
        
        
        mentionsdf = getMentionsResults(candidateMatchesDF)
        
        if mentionsDF is None:
            mentionsDF = mentionsdf
        else:
            mentionsDF = mentionsDF.append(mentionsdf)
    
        if citationsDF is None:
            citationsDF = resDf
        else:
            if len(resDf) < 1:
                print("No dataset matched mentions for file : ",file)
            else:
                citationsDF = citationsDF.append(resDf)
        print()   
    return citationsDF,mentionsDF

##evaluation function
def evaluatePipeline(samplePublications,Th,Ts,Tgs):
    resultsDF,_ = runPipeLine(samplePublications,max_seq_len = 66, hit_th = Th, \
                            sim_th = Ts,group_sim_th = Tgs)
    if resultsDF is None:
        return {'Th':Th, 'Ts':Ts ,'Tgs':Tgs, 'recall': 0, 'precision': 0, 'fscore': 0}
    trueLabels = getTrueLabels(samplePublications)
    recall,precision,fscore = evaluationMetrics(trueLabels,resultsDF)
    return {'Th':Th, 'Ts':Ts ,'Tgs':Tgs, 'recall': recall, 'precision': precision, 'fscore': fscore}

def get_publications():
    pubText = load_doc(PUBLICATIONS_JSON_FILE)
    pubJson = json.loads(pubText)
    return pubJson

publications = get_publications()
dataSetIds,dataSetTitles = loadDataSetTitlesAndIds(DATSETS_JSON_FILE)
dataset_name_to_Id = dict(zip(dataSetTitles, dataSetIds))

cnnVocab = loadVocab(CNN_VOCAB_FILE)
cnnTokenizer = load_Keras_Tokenizer_CNN(CNN_TOKENIZER_File)
model = load_CNN_Sentece_Classifier(CNN_MODEL_FILE, CNN_MODEL_WEIGHTS_FILE)

dataset_vocab = loadVocab(DATASET_VOCAB_FILE)
datasetlines = getDataSetProcessedLines()
vectorizerDataset = getDatasetNgramVectorizer(datasetlines)
dataset_Ngram = vectorizerDataset.transform(datasetlines)
print('DataSet TFID shape: ',dataset_Ngram.shape)   

sampleSize = 10
sampleIndex = np.random.randint(0,len(publications),sampleSize)
samplePublications = [t for i,t in enumerate(publications) if i in sampleIndex]
len(samplePublications)

resultsDF,matchDF = runPipeLine(samplePublications,max_seq_len = 66, hit_th = 0.8, \
                            sim_th = 0.2,group_sim_th = 0.8)

save_citations(resultsDF)
save_mentions(matchDF)


##########################################################--Research fields--################################################################3
def fields_clean_doc(doc):
    #
    doc = doc.replace('(general)','')
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # make lower case
    tokens = [word.lower() for word in tokens]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter special words
    special_words = ['fieldaltlabel','fieldid','fieldlabel']
    tokens = [w for w in tokens if not w in special_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):
	# load the doc
	doc = load_doc(filename)
	# clean doc
	tokens = fields_clean_doc(doc)
	# filter by vocab
	tokens = [w for w in tokens if w in vocab]
	return ' '.join(tokens)

# load all docs in a directory
def fields_process_docs(publications, vocab):
    lines = list()
    for pub in publications:
        path = TEXT_DIRECTORY+pub['text_file_name']
        line = doc_to_line(path, vocab)
        lines.append(line)
    return lines

def getFieldsSimMatrix(field_lines,pub_lines):
    # create the tokenizer
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    # fit the tokenizer on the documents
    fieldTokenizer = vectorizer.fit(field_lines)
    # encode training data set
    fieldsNgram = fieldTokenizer.transform(field_lines)
    #print(fieldsNgram.shape)
    # encode training data set
    pubNgram = fieldTokenizer.transform(pub_lines)
    #print(doc_bow.shape)
    data_sim = cosine_similarity(pubNgram,fieldsNgram,dense_output=True)
    #print(data_sim.shape)
    #group_sim_scores = min_max_scale(data_sim.reshape(data_sim.shape[1]))
    return data_sim

def getSageFields(publications):
    text = load_doc(SAGE_FIELDS_FILE)
    sage_fields_json = json.loads(text)
    fields = list(sage_fields_json.keys())

    sageVocab = load_doc(SAGE_VOCAB_FILE).split()
    processed_lines = fields_process_docs(publications, sageVocab)
    field_lines = load_doc(SAGE_FIELDS_LINES).split('\n')
    sim =getFieldsSimMatrix(field_lines, processed_lines)
    fieldLabels = [fields[np.argmax(c_s)] for c_s in sim]

    SubFieldLabels = []
    for i in range(len(processed_lines)):
        subFieldJson = sage_fields_json[fieldLabels[i]]
        subFields = list(subFieldJson.keys())
        subLines = []
        for sf in subFields:
            subLines.append(' '.join(clean_doc(str(subFieldJson[sf]))))
        subSim =getFieldsSimMatrix(subLines, [processed_lines[i]])
        label = subFields[np.argmax(subSim[0])]
        SubFieldLabels.append(label)

    finalLabels = [f+" : "+s for f,s in zip(fieldLabels,SubFieldLabels)]
    pubIds = [p['publication_id'] for p in publications]
    score = np.ones(len(publications))
    return pd.DataFrame({'publication_id': pubIds, 'research_field': finalLabels, 'score': score}).to_dict('records')

sageFieldsJson = getSageFields(samplePublications)
with open(OUTPUT_DIRECTORY+'research_fields.json', 'w') as fp:
        json.dump(sageFieldsJson, fp)
