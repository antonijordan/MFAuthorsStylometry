import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

for token in doc:
    print(token.text, token.dep_, token.dep, token.head)


import stanza
nlp = stanza.Pipeline('en', processors='tokenize, mwt, pos, lemma, depparse')
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

# this looks better
dict = {"word_id": [], }
for sent in doc.sentences:
    for word in sent.words:
        print(word.id, word.text, word.head, word.deprel)
        if word.head > 0:
            print(sent.words[word.head-1].text)
        else:
            print("root")

# word id - position in the sentence
# word.text - the actual word
# word.head - position of the head in the sentence
# word.deprel - the kind of dependency relation

# https://stanfordnlp.github.io/stanza/depparse.html