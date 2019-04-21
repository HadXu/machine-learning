# chapter2

NLP(Natural language processing)和CL(computational linguistics)是两种对人类语言的研究工具。NLP是来解决人与人之间的问题(such as information extraction, automatic speech recognition, machine translation, sentiment ana
lysis, question answering, and summarization), 而CL是解决理解人类语言的本质。(How do we understand language? How do we produc
e language? How do we learn languages? What relationships do languages have with one another?)

## Spacy工具

我们一般使用Spacy进行处理。

1. Tokenizing text
```angular2html
import spacy
nlp = spacy.load('en')
text = "Mary, don’t slap the green witch"
print([str(token) for token >in nlp(text.lower())])
```
2. Generating n-grams from text
```angular2html
def n_grams(text, n):
    '''
    takes tokens or text, returns a list of n-grams
    '''
    return [text[i:i+n] for i in range(len(text)-n+1)]

cleaned = ['mary', ',', "n't", 'slap', 'green', 'witch', '.']
print(n_grams(cleaned, 3))
```
3. Lemmatization: reducing words to their root forms
```angular2html
import spacy
nlp = spacy.load('en')
doc = nlp(u"he was running late")
for token in doc:
    print('{} --> {}'.format(token, token.lemma_))
```
4. Part-of-speech tagging
```angular2html
import spacy
nlp = spacy.load('en')
doc = nlp(u"Mary slapped the green witch.")
for token in doc:
    print('{} - {}'.format(token, token.pos_))
```