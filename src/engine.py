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
    """
    A Named Entity Recognition (NER) processing engine that supports multiple models and provides various utilities for text preprocessing, entity extraction, and visualization.

    Attributes:
        devMode (bool): A flag to enable or disable development mode for additional logging and debugging.
        nlp (spacy.Language or transformers.pipelines.Pipeline): The NLP model used for entity recognition.
        text (str): The text to be processed.
        entities (list): A list of extracted entities.
        docs (spacy.tokens.Doc or list): The processed document.
        graph_html (str): The HTML representation of the entity graph.
    """

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
        """
        Preprocess the input text by performing the following steps:
        1. If no text is provided, use the instance's text attribute.
        2. Convert HTML content to plain text.
        3. Remove all non-alphanumeric characters except whitespace.
        4. Remove stopwords from the text.

        Args:
            text (str, optional): The text to preprocess. Defaults to None.

        Returns:
            str: The preprocessed text.
        """

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
        """
        Adds the 'entityfishing' component to the NLP pipeline with specified configuration.
        This method configures and adds the 'entityfishing' component to the NLP pipeline.
        The 'entityfishing' component is used for named entity recognition (NER) and linking.
        It connects to an external API service for entity recognition.

        Configuration:
        - language: The language for entity recognition, set to 'en' (English).
        - api_ef_base: The base URL for the entityfishing API service.

        Note:
        - There is a TODO to add a 'spancat' component for probability scoring in the future.
        """

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
        """
        Adds a node to the graph `g` with the specified `node_name` and `node_type`.
        This method adds a node with a box shape and a title corresponding to the `node_type`.
        It also attempts to add another node with an ellipse shape for the `node_type`.
        Finally, it creates an edge between the `node_type` node and the `node_name` node.

        Args:
        g (networkx.Graph): The graph to which the nodes and edge will be added.
        node_name (str): The name of the node to be added.
        node_type (str): The type of the node, used for labeling and coloring.

        Exceptions:
        Raises exception if there is an error adding the `node_type` node, it will be caught and printed.
        """

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
        """
        Converts the entities to a graph representation and generates an HTML file.

        Args:
            height (str, optional): The height of the graph. Defaults to "400px".
            width (str, optional): The width of the graph. Defaults to "50%".

        Returns:
            None: The generated HTML is stored in the `graph_html` attribute.
        """

        g = Network(height=height, width=width, heading="")

        for data in self.entities:
            self.add_node(g, data[0], data[1])

        self.graph_html = g.generate_html("example.html")

    def postprocess_and_print(self, doc):
        """
        Processes the entities in the given document and prints them if in development mode.

        Args:
            doc (spacy.tokens.Doc): The document containing entities to be processed.

        Returns:
            list: A list of tuples where each tuple contains the text and label of an entity.
        """

        entities = [(ent.text, ent.label_) for ent in doc.ents]

        if self.devMode:
            for i in entities:
                print(i)

        self.entities = entities
        return entities

    def process_with_entityFishing(self, text):
        """
        Processes the given text using the entityFishing NLP model.
        This method preprocesses the input text and then applies the entityFishing
        NLP model to extract entities from the text. The processed document is
        stored in the instance variable `self.docs`.

        Args:
            text (str): The input text to be processed.

        Returns:
            doc: The processed document containing extracted entities.
        """

        doc = self.nlp(self.preprocess(text))
        self.docs = doc
        return doc

    def render_entities(self, doc=None, jupyter=False, selected_options=None):
        """
        Renders named entities in the given document using spaCy's displacy visualizer.

        Args:
            doc (spacy.tokens.Doc, optional): The document containing the entities to render.
                                               If None, the method will use `self.docs`. Defaults to None.
            jupyter (bool, optional): If True, renders the entities in a Jupyter notebook. Defaults to False.
            selected_options (list, optional): A list of entity types to render. Defaults to a predefined list of entity types.

        Returns:
            str: The HTML or Jupyter rendering of the entities.
        """
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
        """
        Processes the input text through the entity extraction pipeline and converts the results to a graph.

        Args:
            html_width (str, optional): The width of the HTML representation of the graph. Defaults to "100%".
            html_height (str, optional): The height of the HTML representation of the graph. Defaults to "40vh".
            text (str, optional): The input text to be processed. If not provided, the existing text attribute will be used.

        Returns:
            None
        """
        if text is not None:
            self.text = text

        doc = self.process_with_entityFishing(self.text)
        ent = self.postprocess_and_print(doc)
        self._convert_to_graph(height=html_height, width=html_width)

        if self.devMode:
            print(ent)
