import re
import pandas as pd
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import nltk
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

nltk.download("stopwords")


class NERProcessingEngine:
    def __init__(self, filepath=None, model="spacy"):
        if model == "spacy":
            self._load_spacy_model()
        elif model == "hf":
            self._load_HF_NER_model()

        self.devMode = True

        if filepath:
            self.load_file_by_path(filepath)

    def _load_spacy_model(self):
        self.nlp = spacy.load("en_core_web_sm")
        self._add_entityFishing()

    def _load_HF_NER_model(self):
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

        self.nlp = pipeline("ner", model=model, tokenizer=tokenizer)

    def load_file_by_path(self, filepath):
        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()

        self.text = text

    def preprocess(self, text=None):
        if text is None:
            text = self.text

        # Lower case the text
        # text = text.lower()

        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

        stop_words = set(stopwords.words("english"))
        text = " ".join(word for word in text.split() if word not in stop_words)

        return text

    def _add_entityFishing(self):
        self.nlp.add_pipe(
            "entityfishing",
            config={
                "language": "en",
                "api_ef_base": "http://nerd.huma-num.fr/nerd/service",
            },
        )
        # nlp.add_pipe("spancat")

    # def parse(self, text):
    #     print(text)

    #     ner_results = self.nlp(text)
    #     print(ner_results)

    #     self.df = pd.DataFrame(ner_results)
    #     print(self.df)

    def postprocess_and_print(self, doc):
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        if self.devMode:
            for i in entities:
                print(i)

        return entities

    def process_with_entityFishing(self, text):
        return self.nlp(self.preprocess(text))

    def pipeline(self, text=None):
        if text is not None:
            self.text = text
        doc = self.process_with_entityFishing(self.text)
        ent = self.postprocess_and_print(doc)
        self.entities = ent

        if self.devMode:
            print(ent)
