import re
import pandas as pd
from bs4 import BeautifulSoup

import nltk
import spacy
from spacy import displacy
from nltk.corpus import stopwords
from pyvis.network import Network

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

nltk.download("stopwords")


class NERProcessingEngine:
    def __init__(self, filepath=None, model="spacy", devMode=False):
        if model == "spacy":
            self._load_spacy_model()
        elif model == "hf":
            self._load_HF_NER_model()

        self.devMode = devMode

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

        # TODO: add spancat for probability score
        # nlp.add_pipe("spancat")

    def _get_colormap(self):
        return {
            "CARDINAL": "#FF0000",
            "DATE": "#0000FF",
            "EVENT": "#008000",
            "FAC": "#800080",
            "GPE": "#FFA500",
            "LANGUAGE": "#FFC0CB",
            "LAW": "#A52A2A",
            "LOC": "#00FFFF",
            "MONEY": "#FFFF00",
            "NORP": "#FF00FF",
            "ORDINAL": "#00FF00",
            "ORG": "#008080",
            "PERCENT": "#FFD700",
            "PERSON": "#000000",
            "PRODUCT": "#C0C0C0",
            "QUANTITY": "#000080",
            "TIME": "#800000",
            "WORK_OF_ART": "#808000",
        }

    def add_node(self, g, node_name, node_type):
        g.add_node(
            node_name,
            label=node_name,
            shape="box",
            title=node_type,
            color=self._get_colormap()[node_type],
        )

        try:
            g.add_node(
                node_type,
                label=node_type,
                shape="ellipse",
                color=self._get_colormap()[node_type],
            )
        except Exception as e:
            print(e)

        g.add_edge(node_type, node_name)

    def _convert_to_graph(self, height="400px", width="50%"):
        g = Network(height=height, width=width, heading="")

        for data in self.entities:
            self.add_node(g, data[0], data[1])

        self.graph_html = g.generate_html("example.html")

    def postprocess_and_print(self, doc):
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        if self.devMode:
            for i in entities:
                print(i)

        self.entities = entities
        return entities

    def process_with_entityFishing(self, text):
        doc = self.nlp(self.preprocess(text))
        self.docs = doc
        return doc

    def render_entities(self, doc=None, jupyter=False, selected_options=None):
        selected_options = [
            "CARDINAL",
            "DATE",
            "EVENT",
            "FAC",
            "GPE",
            "LANGUAGE",
            "LAW",
            "LOC",
            "MONEY",
            "NORP",
            "ORDINAL",
            "ORG",
            "PERCENT",
            "PERSON",
            "PRODUCT",
            "QUANTITY",
            "TIME",
            "WORK_OF_ART",
        ]
        options = {"ents": selected_options}  # , "colors": get_entity_colors()}

        if doc is None:
            doc = self.docs
        return displacy.render(doc, style="ent", jupyter=jupyter, options=options)

    def pipeline(
        self,
        html_width="100%",
        html_height="40vh",
        text=None,
    ):
        if text is not None:
            self.text = text

        doc = self.process_with_entityFishing(self.text)
        ent = self.postprocess_and_print(doc)
        self._convert_to_graph(height=html_height, width=html_width)

        if self.devMode:
            print(ent)
