# Entity Extraction and Relationship Modeling

This project demonstrates an approach to extract entities (e.g., people, organizations, locations) and identify relationships within text/Markdown files. It includes a development endpoint using a Python-based framework and visualizes the extracted nodes and their relationships.

## Tools and Methodologies

- **Regular Expressions (re)**: Used for text preprocessing and cleaning.
- **Pandas (pd)**: Utilized for data manipulation and analysis.
- **BeautifulSoup**: Employed for parsing HTML content.
- **NLTK**: Used for natural language processing tasks, including stopwords removal.
- **spaCy**: A powerful NLP library used for named entity recognition (NER) and visualization.
- **PyVis**: A library for visualizing networks in Python.
- **Hugging Face Transformers**: Utilized for state-of-the-art NER models.

## Installation

1. Clone the repository:
    ```bash
    git clone <repository_url>
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Initialize the `NERProcessingEngine` with the desired model (`spacy` or `hf`):
    ```python
    from src.engine import NERProcessingEngine

    engine = NERProcessingEngine(filepath="path/to/text/file", model="spacy", devMode=True)
    ```
2. Process the text and visualize entities:
    ```python
    engine.pipeline()
    ```

## Development Endpoint

The Python backend is written in Flask, and the endpoint is exposed at `/endpoint`. To start the server, use the following command.

```python
python app.py
```

The endpoint can be tested using test script prepared in `test/submit_post_request.py` It details payload signature and currently only accepts text form. 


## Example file

Refer to the provided Markdown file (`ProjectPhoenixPlan.md`) for an example input to test the program.