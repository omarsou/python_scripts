from utils.lazy_module import Lazy
import torchvision


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

def _loadMaskrcnnResnet():
    import torchvision
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model
maskrcnn_resnet = Lazy(_loadMaskrcnnResnet)
