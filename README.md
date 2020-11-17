# stanza-batch
Is a batching utility for [Stanza](https://github.com/stanfordnlp/stanza) making processing documents/texts with Stanza quicker and easier. The current recommendation for batching by [Stanza is to concatenate documents together with each document separated by a blank line (`\n\n`)](https://github.com/stanfordnlp/stanza#batching-to-maximize-pipeline-speed). This way of batching has one main drawback:

1. The return of processing this document is one Stanza Document with lots of sentences, thus you don't know where one document ends and another starts, easily.

This batching utility solves this problem, when given a list of documents, it will return a list of corresponding processed Stanza documents. Below we show a comparison of the current Stanza batching system and how batching works with this utility:

```python
import stanza
from stanza.models.common.doc import Document
# Documents to process
document_1 = 'Hello how are you\n\nI am good thank you'
document_2 = 'This is a different document'
# Create the document batch
batch_document = '\n\n'.join([document_1, document_2])
# stanza model
nlp = stanza.Pipeline(lang="en", processors="tokenize",)
stanza_batch = nlp(batch_document)
assert isinstance(stanza_batch, Document)
stanza_document = 'Hello how are you\n\nI am good thank you\n\nThis is a different document'
assert stanza_batch.text == stanza_document
assert len(stanza_batch.sentences) == 3

# Using Stanza Batch
from typing import List
from stanza_batch import batch
stanza_documents: List[Document] = []
for document in batch([document_1, document_2], nlp, batch_size=32): # Default batch size is 32
    stanza_documents.append(document)
assert len(stanza_documents) == 2 # 2 Documents after processing
# Each document contains the same raw text after processing
assert stanza_documents[0].text == document_1 
assert stanza_documents[1].text == document_2
# Each document contains the expected number of sentences
assert len(stanza_documents[0].sentences) == 2
assert len(stanza_documents[1].sentences) == 1
```

As we can see above the new `batch` function yields corresponding processed documents from the given iterable `[document_1, document_2]`. Compared to the original version which only yields one processed document for all the documents in the iterable.

## Installation

Requires Python 3.6.1 or later. As the package depends on [Stanza]() which also depends on Pytorch we recommend that you install the version of Pytorch that suits your setup first (e.g. CPU or GPU Pytorch and then if GPU a specific CUDA version).

``` bash
pip install .
```

Currently it has only been tested for Stanza version `1.1.1`.

## Edge cases

The way `stanza-batch` performs batching it does so by making use of the `\n\n` batching approaching, but it keeps track of the documents given to the batch process. However by making use of the `\n\n` batching approach and other assumptions it does come with some edge cases. All of these edge cases will mean that the length of the document in characters will be different as whitespace is removed from the documents, but content characters will not be removed:

### Removal of whitespace at the start and end of documents

As we only keep track of token offsets any whitespace at the start or end of document will be removed:
```python
import stanza
from stanza_batch import batch
# stanza model
nlp = stanza.Pipeline(lang="en", processors="tokenize",)
document_1 = '\n  Hello how are you\n\nI am good thank you  \n'
stanza_document = [doc for doc in batch([document_1], nlp)][0]
assert stanza_document.text == 'Hello how are you\n\nI am good thank you'
```

### Swapping of whitespace characters

The batching approach does not actually split on `\n\n` it actually makes use of the following regex `\n\s*\n`, to ensure that we can keep track of which Stanza sentence belongs to which document we split each document with this regex and replace that whitespace with `\n\n`. Therefore this means that if you had a document that contains `\n  \n` this will be replaced with `\n\n`:
```python
import stanza
from stanza_batch import batch
# stanza model
nlp = stanza.Pipeline(lang="en", processors="tokenize",)
document_1 = 'Hello how are you\n \n \nI am good thank you'
stanza_document = [doc for doc in batch([document_1], nlp)][0]
assert stanza_document.text == 'Hello how are you\n\nI am good thank you'
```

## Handling Out Of Memory (OOM) errors

When batching for inference you normally want the largest possible batch size without causing an OOM (for either CPU or GPU memory). A package that I found useful for this is the [toma package](https://github.com/BlackHC/toma). The [`toma.simple.batch` method](https://github.com/BlackHC/toma/blob/master/toma/__init__.py#L25) wraps a function and if a RuntimeError is caused it will reduce the batch size by half and re-run the function until it finishes or it causes another RuntimeError and then the process is repeated. Example of how to use it with this project, in this example we use part of the Jane Austin Emma book as the text data which can be found in the test data [./tests/data/jane_austin_emma_data.txt](./tests/data/jane_austin_emma_data.txt):
```python
from typing import List
from pathlib import Path
import stanza
from stanza.models.common.doc import Document
import toma

from stanza_batch import batch

# toma requires the first argument of the method to be the batch size
def run_batch(batch_size: int, stanza_nlp: stanza.Pipeline, data: List[str]
              ) -> List[Document]:
    # So that we can see what the batch size changes to.
    print(batch_size)
    return [doc for doc in batch(data, stanza_nlp, batch_size=batch_size)]

emma_fp = Path(__file__, '..', 'tests', 'data', 'jane_austin_emma_data.txt').resolve()
jane_austin_data: List[str] = []
with emma_fp.open('r') as emma_file:
    jane_austin_data = [line for line in emma_file]
# Make the text much larger to cause OOM
jane_austin_data = jane_austin_data * 5
assert len(jane_austin_data) == 2450
# We set the initial batchsize as 500 and add POS and NER tagging
nlp = stanza.Pipeline(lang="en", processors="tokenize,pos,ner", 
                      tokenize_batch_size=500,
                      ner_batch_size=500,)

processed_documents: List[Document] = toma.simple.batch(run_batch, 500, 
                                                        nlp, jane_austin_data)
assert len(processed_documents) == 2450
```
Running this code on my local machine which had 16GB RAM and Nvidia 1060 6GB GPU reduces the batch size from 500 to 62 and took around 102 seconds to run. If I had ran this code knowing 62 is close to the largest batch size that my computer can mange the code would have taken 90 seconds (keeping the ner_batch_size=500 but all others to 62), showing the time/processing cost of causing an OOM exception.

## Developing/Testing

This code base uses `flake8`, `mypy`, and `black` to ensure that the format of the code is consistent. The `flake8` defaults ([./.flake8](./.flake8)) are the same as those in the [Spacy project](https://github.com/explosion/spaCy/blob/master/setup.cfg#L94) and the `black` defaults follow those as well. The `mypy` requirements is mainly here due to my own preference on having type hints in the code base.

To install all of these requirements:
```bash
pip install -r dev-requirements.txt
```

To use black, flake8, mypy, and pytest use the following commands:
``` bash
black --line-length 80 .
flake8 .
mypy
python -m pytest --cov=stanza_batch --cov-report term-missing
```

The flake8, mypy, and pytest have to pass whereby the pytest test coverage should be 100% for a pull request to be accepted. If these requirements are not met in your pull request we will work with you to resolve any issues, so please do not get put off creating a pull request if you cannot pass any/all of these requirements.