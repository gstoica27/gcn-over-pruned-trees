import stanza

stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize,ner', tokenize_pretokenized=True)
tokens = ' '.join(['John', 'dropped', 'out', 'of', 'NYU', '.'])

