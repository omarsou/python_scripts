from utils.lazy_object_load import sentence_tokenize, nlp_stanford
import re

replacements = [(r'(?<!\n)\n(?![\n\t])', ' '),
                (r'http\S+', ''),  # remove link http
                (r'www\S+', ''),  # remove link wwww
                (r'\S*@\S*\s?', '@')]  # replace mail by @ # Can be customized
sentences_to_delete = ['##- Please type your reply above this line -##', '---------- Forwarded message ---------',
                       '\t', '  ']  # Can be customized


class MailParser:
    def __init__(self, threshold=0.960, limit_not_taken=2):
        self.threshold = threshold  # Can be customized
        self.limit_not_taken = limit_not_taken  # Can be customized

    def parse_mail(self, txt):
        """Parse the mail to get only the body (which is the main part of the mail) of the mail.

        Parameters
        ----------
        txt : str
              Represent the email

        Returns
        -------
        parsed_mail : str
                      Parsed email.
        """

        preproc_mail = self.preprocess(txt)
        lines = self.mail2lines(preproc_mail)
        try:
            parsed_mail = self.generate_body_mail(lines=lines, threshold=self.threshold)
        except KeyError:
            return float('nan')
        return parsed_mail

    def generate_body_mail(self, lines, threshold):
        """iterate through lines. if the line is a true sentence,
           add it to final_sentence.

           if probability(sentence to be a true sentence) < threshold, then it is a true sentence.

        Parameters
        ----------
        lines : list
                    Represents the list of block of text in the email.
        threshold: float
                   Lower thresholds will result in more false positives.

        Returns
        -------
        final_sentence : str
                         Parsed email block.
        """

        delete_useless = False
        sentence_not_taken = 0
        final_sentence = ''
        for line in lines:
            for sent in sentence_tokenize.value.tokenize(line):
                if len(sent) < 5:
                    continue
                if self.compute_prob_not_sentence(sent) < threshold:
                    delete_useless = True
                    final_sentence += sent + ' '
                    sentence_not_taken = 0
                    continue
                if delete_useless:
                    sentence_not_taken += 1
                if delete_useless and sentence_not_taken >= self.limit_not_taken:
                    break
            if delete_useless and sentence_not_taken >= self.limit_not_taken:
                break
        return final_sentence

    @staticmethod
    def preprocess(txt):
        """some preprocessing"""
        text = txt.replace('\n \n', '\n\n').replace(':\n', '\n\n').replace(',\n', '\n\n')  # Can be customized
        for bad in sentences_to_delete:
            text = text.replace(bad, '')  # Can be customized
        for old, new in replacements:
            text = re.sub(old, new, text)  # Can be customized
        return text

    @staticmethod
    def mail2lines(corpus):
        """split the mail into list of lines.
        """
        return corpus.strip().split('\n')

    @staticmethod
    def compute_prob_not_sentence(sentence):
        """Calculate probability that the sentence is a not a TRUE sentence.

        Parameters
        ----------
        sentence : str
            Line in email block.

        Returns
        -------
        probability(sentence to be a true sentence) : float
        """
        doc = nlp_stanford.value(sentence)
        verb_count = 0
        word_count = 0
        for sent in doc.sentences:
            for word in sent.words:
                word_count += 1
                if word.upos in ["VERB", "AUX"]:  # Can be customized
                    if word.xpos in ['VBG']:
                        verb_count += 0.5
                    else:
                        verb_count += 1
        return 1 - verb_count / word_count