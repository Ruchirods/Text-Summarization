import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import PorterStemmer
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sentiment_analysis import SentimentAnalysis
from sklearn.utils.extmath import randomized_svd
import xlwt
import sys
import xml.etree.ElementTree as xml
import math

def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)
"""Performs the preprocessing of text like punctuation removal and 
POS tagging to find the sentences with noun and adjectives and drop others."""
def preprocessing():
    inputfile = open("hotel1.txt", 'r')
    line = inputfile.read()  # Use this to read file content as a stream:
    noofreviews=line.count('\n')
    print "Total no of Reviews:",(noofreviews+1)
    sentence=sent_tokenize(line)
    newsen=[]
    nounchunk=[]
    for sen in sentence:
        sen=sen.lower()
        sen1=""
        sen1=strip_punctuation(sen)
        sen1 = ' '.join([word for word in sen1.split() if word not in stopwords.words("english")])
        newsen.append(sen1)
        taggedsen=nltk.pos_tag(word_tokenize(sen1))
        grammar = "NP: {<DT>?<JJ>*<NN>}"
        cp = nltk.RegexpParser(grammar)
        result = cp.parse(taggedsen)
        for subtree in result.subtrees():
            adjectivecount=0
            nouncount=0
            if subtree.label()=='NP':
                nouncount=nouncount+sum(1 for word,tag in subtree.leaves() if tag.startswith('N'))
                adjectivecount=adjectivecount+sum(1 for word,tag in subtree.leaves() if tag.startswith('J'))
                if nouncount>0 and adjectivecount>0:
                    data=""
                    for word in subtree.leaves():
                        data=data+" "+word[0]
                    nounchunk.append(data)
    return nounchunk

def stopwordsremoval(reviewterm):
    stop_words = set(stopwords.words("english"))
    preprossedfile={}
    ps=PorterStemmer()
    for key,values in reviewterm.iteritems():
        for value in values:
            sen = ""
            for word in word_tokenize(value):
                if word not in stop_words and word not in punctuation:
                    sen=sen+" "+ps.stem(word)
            try:
                preprossedfile[key].append(sen)
            except KeyError:
                preprossedfile[key]=[sen]
    return preprossedfile

def tagging(processeddata):
   taggedfile={}
   for key,values in processeddata.iteritems():
       for value in values:
           count=sum(1 for word, pos in nltk.pos_tag(word_tokenize(value)) if pos.startswith('N') or pos.startswith('J'))
           if count>0:
               try:
                   taggedfile[key].append(value)
               except KeyError:
                   taggedfile[key] = [value]

   return taggedfile

def CountVec(nounchunks,noofsentence):
    vectorizer = CountVectorizer(min_df=1,stop_words = 'english')
    A = vectorizer.fit_transform(nounchunks)
    A = A.transpose()
    A = A.toarray()
    print A
    U, S, VT = randomized_svd(A, n_components=100,
                              n_iter=5,
                              random_state=None)
    print VT.shape, A.shape, U.shape, S.shape
    C=np.matmul(U,S)
    return U,S,VT,C

def square(list):
    sum=0
    for i in list:
        sum=sum+i*i
    return sum

"""Performs singular value decomposition on the preprocessed document and finds the the summary using VT matrix"""
"""Also computes the similarity between the input and summary generated"""
def SingularValueDecomposition(nounchunks,noofsentence):
    summary=''
    finalsummary=[]
    file=open("TxtSummary.txt","w")
    book = xlwt.Workbook(encoding="utf-8")
    sheet1 = book.add_sheet("Summary")
    sheet1.write(0, 0, "NounChunks")
    sheet1.write(0, 1, "RSV")
    if len(nounchunks)<noofsentence:
        for chunks in nounchunks:
            summary=summary+" "+chunks
    else:
        U,S,VT,Input=CountVec(nounchunks,noofsentence)
        print len(Input)
        Input_dec=sorted(Input,reverse=True)
        #print Input
        VTarray=np.array(VT)
        index=VTarray.argmax(1)
        maxvalue=VT.max(1)
        #print len(nounchunks)
        for i in range(1,noofsentence):
            file.write(nounchunks[index[i-1]]+'\n')
            finalsummary.append(' '+nounchunks[index[i-1]])
            sheet1.write(i,0,nounchunks[index[i-1]])
            sheet1.write(i,1,maxvalue[i-1])
        file.close()

        U,S,VT,Output=CountVec(finalsummary,noofsentence)
        print len(Output)
        Output_dec=sorted(Output,reverse=True)
        Input = Input_dec[:len(Output_dec)]
        simneu=np.dot(Input,Output_dec)
        inputMag=square(Input)
        outputMag=square(Output_dec)
        inputMag=np.sqrt(inputMag)
        outputMag=np.sqrt(outputMag)
        similarity=simneu/(inputMag*outputMag)
        theta=math.acos(similarity)*180/math.pi
        print "Differentce between the input and summary:",theta

"""function to find the sentiment of nounchunks"""
def sentiment(nounchunk):
    sa=SentimentAnalysis()
    positive_sentences, negative_sentences, neutral_sentences = sa.get_sentence_orientation(nounchunk)
    print positive_sentences
    print neutral_sentences
    positive_neutral_sen=positive_sentences+neutral_sentences
    return positive_neutral_sen

def main():
    reload(sys)
    sys.setdefaultencoding('utf-8')
    nounchunk=preprocessing()
    positive_neutral=sentiment(nounchunk)
    print len(positive_neutral)
    SingularValueDecomposition(positive_neutral,15)

if __name__=='__main__':
    main()

