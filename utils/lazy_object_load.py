from utils.lazy_module import Lazy

# Stanford NLP object
def _loadStandford():
    import stanfordnlp
    return stanfordnlp.Pipeline(lang = "en", use_gpu=False, processors="tokenize,pos")
nlp_stanford = Lazy(_loadStandford)

# NLTK sentence_tokenize
def _loadnltkSentencetokenize():
    import nltk.data
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    return tokenizer
sentence_tokenize = Lazy(_loadnltkSentencetokenize)
