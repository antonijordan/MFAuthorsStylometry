import pandas as pd
import re
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from ast import literal_eval
## ładuje model językowy
nlp = spacy.load("en_core_web_sm")

df = pd.read_csv("C:\\Users\\tosik\\OneDrive\\Pulpit\\BA\\dataanalysis\\BA Corpus.csv", encoding = "utf-8")
## wywalam puste wiersze (metoda)
df = df.dropna(subset = "Dialogue")
##print(df)
"""sent = df["Dialogue"][0]
print(sent)
sent = re.sub("\"|'", "", sent)
print(sent)"""

df["Clean_Dialogue"] = df["Dialogue"].apply(lambda x: re.sub(r"\"|'|\n", "", x))
df["Clean_Dialogue"] = df["Dialogue"].apply(lambda x: re.sub(r",|—", " ", x))
df["Clean_Dialogue"] = df["Clean_Dialogue"].apply(lambda x: re.sub(r"\?|\!", ".", x))

df["Clean_Dialogue"] = df["Clean_Dialogue"].apply(lambda x: x.split("."))
df = df.dropna(subset = "Clean_Dialogue")

def tokenize(lst):
    tokenizedtext = []
    for sent in lst:
        words = word_tokenize(sent)
        tokenizedtext.append(words)
    return tokenizedtext

df["Tokenized_Dialogue"] = df["Clean_Dialogue"].apply(lambda x: tokenize(x))

def lemmatization(lst):
    lemmas = []
    for sent in lst:
        sent = nlp(sent)
        sentlem = [token.lemma_ for token in sent if token.text != "," or token.text == " "]
        new_stem_lem = [word for word in sentlem if word != " "]
        lemmas.append(new_stem_lem)
    return lemmas

df["Lemmatized_Dialogue"] = df["Clean_Dialogue"].apply(lambda x: lemmatization(x))

print(df["Tokenized_Dialogue"][1])

##count word number for tokenized and lemmatized, then ttr, then mattr

def count_tokens(lst):
    total_tokens = []
    for sent in lst:
        total_tokens.append(len(sent))
    return total_tokens


df["Token_Count"] = df["Tokenized_Dialogue"].apply(lambda x: count_tokens(x))

def count_lemmas(lst):
    total_lemmas = []
    for sent in lst:
        total_lemmas.append(len(set(sent)))
    return total_lemmas

df["Lemma_Count"] = df["Lemmatized_Dialogue"].apply(lambda x: count_lemmas(x))


##TTR
i = 0

for sent in df["Lemma_Count"].to_list():
    if sum(sent) == 0:
        print(i)
        i += 1
    else:
        i += 1
        
i = 0

for sent in df["Token_Count"].to_list():
    if sum(sent) == 0:
        print(i)
        i += 1
    else:
        i += 1

def ttr(lst_token_count, lst_lemma_count):
    try:
        token_count = sum(lst_token_count)
        lemma_count = sum(lst_lemma_count)
        result = lemma_count/token_count
        return result
    except:
        return 0

df["TTR"] = df.apply(lambda x: ttr(x.Token_Count, x.Lemma_Count), axis = 1)

##print(df["TTR"])

##MATTR

def mattr(lst_token_count, lst_lemma_count):
        res = []
        for i in range(len(lst_token_count)): 
            if lst_lemma_count[i] and lst_token_count[i] > 0:
                res.append(lst_lemma_count[i]/lst_token_count[i])
            else:
                res.append(0)
        res_norm = sum(res)/len(res)
        return res_norm 

    
df["MATTR"] = df.apply(lambda x: mattr(x.Token_Count, x.Lemma_Count), axis = 1)
print(df["MATTR"])

##ADD 
def dep_parse(lst):
    deps_lst = []
    for sent in lst:

        parsed_sent = nlp(sent)
        dep = [(token.i, token.head.i) for token in parsed_sent]
        deps_lst.append(dep)
    return deps_lst

def ADD_sent(lst):
    dist_lst = []
    for sublst in lst:
        loc = []
        for tup in sublst:
             loc.append(abs(tup[1]-tup[0]))
        dist_lst.append(loc)
    return dist_lst

def ADD(lst):
    textlength = len(lst)
    distances = []
    for sent in lst:
        distances.append(sum(sent))
    return sum(distances)/ textlength

df["Dep_Indices"] = df["Clean_Dialogue"].apply(lambda x: dep_parse(x))
##print(df["Dep_Indices"][0])

df["ADD_sent"] = df["Dep_Indices"].apply(lambda x: ADD_sent(x))
##print(df["ADD_sent"][0])

df["ADD"] = df["ADD_sent"].apply(lambda x: ADD(x))
##print(df["ADD"][0])

df.to_csv("C:\\Users\\tosik\\OneDrive\\Pulpit\\BA\\dataanalysis\\finish\\finaldata.csv", index=False)