from pathlib import Path
from typing import List

import pytest
import stanza
from stanza.models.common.doc import Document

import stanza_batch

EXAMPLE_ONE = "\nHello how are you\n"  # One sentence long
EXAMPLE_TWO = "Hello how are you. Great Thanks\n \n\nSomething else"  # Three sentences in one document
EXAMPLE_THREE = "Another day. Hello how are you. Today is great."  # Three sentences without \n sentence breaker
# Test multiple \n\n break points in one text
EXAMPLE_FOUR = "\nHello how are you. Great Thanks\n \n\nSomething else\n     \n\nAnother test\n"


def test_version() -> None:
    assert isinstance(stanza_batch.__version__, str)


def test_misc_in_change_offsets() -> None:
    """
    Tests that we check if "misc" exists in as a key in token within the
    change_offsets functions. The change_offsets functions are within
    the combine_stanza_documents and _batch_to_documents functions.
    """
    # For the _batch_to_documents function
    stanza.download("cs", processors="tokenize,mwt")
    nlp = stanza.Pipeline(lang="cs", processors="tokenize,mwt", use_gpu=False)
    sent = "Požádal, aby mu vyhověli."
    correct_tokens = [
        [{"id": 1, "text": "Požádal", "misc": "start_char=0|end_char=7"}],
        [{"id": 2, "text": ",", "misc": "start_char=7|end_char=8"}],
        [
            {"id": (3, 4), "text": "aby", "misc": "start_char=9|end_char=12"},
            {"id": 3, "text": "aby"},
            {"id": 4, "text": "by"},
        ],
        [{"id": 5, "text": "mu", "misc": "start_char=13|end_char=15"}],
        [{"id": 6, "text": "vyhověli", "misc": "start_char=16|end_char=24"}],
        [{"id": 7, "text": ".", "misc": "start_char=24|end_char=25"}],
    ]
    docs = list(stanza_batch.batch([sent], nlp))
    assert len(docs) == 1
    for index, token in enumerate(docs[0].iter_tokens()):
        correct_token = correct_tokens[index]
        temp_token = token.to_dict()
        assert len(temp_token) == len(correct_token)
        assert temp_token == correct_token
    # For the combine_stanza_documents function
    doc = stanza_batch.combine_stanza_documents(docs)
    for index, token in enumerate(doc.iter_tokens()):
        correct_token = correct_tokens[index]
        temp_token = token.to_dict()
        assert len(temp_token) == len(correct_token)
        assert temp_token == correct_token


def test__stanza_batch() -> None:
    count = 0
    for batch, offsets, document_index in stanza_batch._stanza_batch(
        [EXAMPLE_ONE]
    ):
        count += 1
        assert document_index == [0]
        assert offsets == [19]
        assert batch == EXAMPLE_ONE + "\n\n"
    assert count == 1

    # Test to ensure documents split into two for the document index.
    count = 0
    for batch, offsets, document_index in stanza_batch._stanza_batch(
        [EXAMPLE_ONE, EXAMPLE_TWO, EXAMPLE_THREE]
    ):
        count += 1
        assert document_index == [0, 1, 1, 2]
        assert offsets == [19, 52, 68, 117]
        # Example two gets split into two sentences within a batch due to how
        # stanza splits sentence with the NEWLINE_WHITESPACE_RE regex.
        example_2_0 = "Hello how are you. Great Thanks"
        example_2_1 = "Something else"
        assert batch == (
            EXAMPLE_ONE
            + "\n\n"
            + example_2_0
            + "\n\n"
            + example_2_1
            + "\n\n"
            + EXAMPLE_THREE
            + "\n\n"
        )
    assert count == 1

    # Test when the batch size is 2 and we have 5 examples to process
    document_index_dict = {0: [0, 1], 1: [1, 2], 2: [3]}
    offsets_dict = {0: [19, 52], 1: [14, 63], 2: [19]}
    batch_dict = {
        0: (EXAMPLE_ONE + "\n\n" + "Hello how are you. Great Thanks" + "\n\n"),
        1: "Something else" + "\n\n" + EXAMPLE_THREE + "\n\n",
        2: EXAMPLE_ONE + "\n\n",
    }
    documents = [EXAMPLE_ONE, EXAMPLE_TWO, EXAMPLE_THREE, EXAMPLE_ONE]
    count = 0
    for index, b_o_d in enumerate(
        stanza_batch._stanza_batch(documents, batch_size=2)
    ):
        count += 1
        batch, offsets, document_index = b_o_d
        assert document_index == document_index_dict[index]
        assert offsets == offsets_dict[index]
        assert batch == batch_dict[index]
    assert count == len(document_index_dict)
    # Ensure ValueError occurs when no text is exist
    with pytest.raises(ValueError):
        [_ for _ in stanza_batch._stanza_batch([EXAMPLE_ONE, "\n\n"])]


def test__batch_to_documents() -> None:
    stanza.download("en", processors="tokenize")
    nlp = stanza.Pipeline(lang="en", processors="tokenize", use_gpu=False)
    # One sample
    count = 0
    for batch, offsets, document_index in stanza_batch._stanza_batch(
        [EXAMPLE_ONE]
    ):
        count += 1
        processed_batch = nlp(batch)
        documents, indexes = stanza_batch._batch_to_documents(
            processed_batch, offsets, document_index
        )
        assert len(documents) == 1
        assert indexes == [0]
        # This process removes the \n either side of the string
        assert documents[0].text == "Hello how are you"
        assert documents[0].sentences[0].tokens[0].start_char == 0
        assert documents[0].sentences[0].tokens[-1].end_char == 17
        document_text = documents[0].text
        for sentence in documents[0].sentences:
            for token in sentence.tokens:
                assert (
                    document_text[token.start_char : token.end_char]
                    == token.text
                )
    assert count == 1

    # One sample where the sample is split into three due to `\n\n` in the
    # middle of the string.
    count = 0
    for batch, offsets, document_index in stanza_batch._stanza_batch(
        [EXAMPLE_FOUR]
    ):
        count += 1
        processed_batch = nlp(batch)
        documents, indexes = stanza_batch._batch_to_documents(
            processed_batch, offsets, document_index
        )
        assert len(documents) == 1
        assert indexes == [0]
        # This process removes the `\n \n\n` and adds `\n\n` in its place.
        assert (
            documents[0].text
            == "Hello how are you. Great Thanks\n\nSomething else\n\nAnother test"
        )
        assert documents[0].sentences[0].tokens[0].start_char == 0
        assert documents[0].sentences[-1].tokens[-1].end_char == 61
        document_text = documents[0].text
        for sentence in documents[0].sentences:
            for token in sentence.tokens:
                assert (
                    document_text[token.start_char : token.end_char]
                    == token.text
                )
    assert count == 1

    # Multiple samples
    document_count = {0: 2, 1: 1, 2: 1}
    text_dict = {
        0: {0: EXAMPLE_ONE.strip(), 1: EXAMPLE_THREE.strip()},
        1: {0: "Hello how are you. Great Thanks\n\nSomething else"},
        2: {0: "Another test"},
    }
    indexes_dict = {0: [0, 1], 1: [2], 2: [2]}
    documents = [EXAMPLE_ONE, EXAMPLE_THREE, EXAMPLE_FOUR]
    count = 0
    for index, b_o_d in enumerate(
        stanza_batch._stanza_batch(documents, batch_size=2)
    ):
        count += 1
        batch, offsets, document_index = b_o_d
        processed_batch = nlp(batch)
        p_i = stanza_batch._batch_to_documents(
            processed_batch, offsets, document_index
        )
        processed_documents, indexes = p_i
        assert len(processed_documents) == document_count[index]
        assert indexes == indexes_dict[index]
        for _document_index, document in enumerate(processed_documents):
            assert document.text == text_dict[index][_document_index]
            for sentence in document.sentences:
                for token in sentence.tokens:
                    assert (
                        document.text[token.start_char : token.end_char]
                        == token.text
                    )
    assert count == len(document_count)

    # Multiple samples in one batch whereby one of the documents is across
    # multiple processed documents
    documents = [EXAMPLE_ONE, EXAMPLE_FOUR, EXAMPLE_THREE]
    count = 0
    for index, b_o_d in enumerate(stanza_batch._stanza_batch(documents)):
        count += 1
        batch, offsets, document_index = b_o_d
        processed_batch = nlp(batch)
        p_i = stanza_batch._batch_to_documents(
            processed_batch, offsets, document_index
        )
        processed_documents, indexes = p_i
        assert len(processed_documents) == 3
        assert indexes == [0, 1, 2]
        assert processed_documents[0].text == EXAMPLE_ONE.strip()
        PROCESSED_EXAMPLE_FOUR_TEXT = (
            "Hello how are you. Great Thanks\n\nSomething else\n\nAnother test"
        )
        assert processed_documents[1].text == PROCESSED_EXAMPLE_FOUR_TEXT
        assert processed_documents[2].text == EXAMPLE_THREE.strip()
        for document in processed_documents:
            for sentence in document.sentences:
                for token in sentence.tokens:
                    assert (
                        document.text[token.start_char : token.end_char]
                        == token.text
                    )
    assert count == 1


def test_combine_stanza_documents() -> None:
    stanza.download("en", processors="tokenize")
    nlp = stanza.Pipeline(lang="en", processors="tokenize", use_gpu=False)
    document_one = nlp(EXAMPLE_ONE)
    document_two = nlp(EXAMPLE_TWO)
    # Two documents combined
    combined_doc = stanza_batch.combine_stanza_documents(
        [document_one, document_two]
    )
    combined_doc_text = combined_doc.text
    assert combined_doc_text == EXAMPLE_ONE + "\n\n" + EXAMPLE_TWO
    sentence_count = 0
    for sentence in combined_doc.sentences:
        sentence_count += 1
        assert sentence.tokens[0].start_char != -1
        for token in sentence.tokens:
            assert (
                combined_doc_text[token.start_char : token.end_char]
                == token.text
            )
    if stanza.__version__ == "1.1.1":
        assert sentence_count == 5
    else:
        assert sentence_count == 4


def test_sentiment_in_sentence() -> None:
    stanza.download("en", processors="tokenize, sentiment")
    nlp = stanza.Pipeline(lang="en", processors="tokenize", use_gpu=False)
    # without sentiment
    for document in stanza_batch.batch([EXAMPLE_ONE], nlp):
        for sentence in document.sentences:
            with pytest.raises(AttributeError):
                sentence.sentiment
    # with sentiment
    nlp = stanza.Pipeline(
        lang="en", processors="tokenize, sentiment", use_gpu=False
    )
    for document in stanza_batch.batch([EXAMPLE_ONE], nlp):
        for sentence in document.sentences:
            assert isinstance(sentence.sentiment, int)


@pytest.mark.parametrize("include_ner", [True, False])
def test_ents_attribute_in_doc_and_sentence(include_ner: bool) -> None:
    entity_document = ["Two entities Alice and Bob"]
    processes = "tokenize"
    if include_ner:
        processes = "tokenize, ner"
    stanza.download("en", processors=processes)
    nlp = stanza.Pipeline(lang="en", processors=processes, use_gpu=False)
    for document in stanza_batch.batch(entity_document, nlp):
        if include_ner:
            assert document.ents
        else:
            assert not document.ents
        for sentence in document.sentences:
            if include_ner:
                assert sentence.ents
            else:
                assert not document.ents


@pytest.mark.parametrize("torch_no_grad", [True, False])
@pytest.mark.parametrize("clear_cache", [True, False])
def test_batch(clear_cache: bool, torch_no_grad: bool) -> None:
    stanza.download("en", processors="tokenize")
    nlp = stanza.Pipeline(
        lang="en", processors="tokenize, sentiment", use_gpu=False
    )
    # One sample
    count = 0
    for document in stanza_batch.batch(
        [EXAMPLE_ONE], nlp, clear_cache=clear_cache, torch_no_grad=torch_no_grad
    ):
        count += 1
        # This process removes the \n either side of the string
        assert document.text == "Hello how are you"
        assert document.sentences[0].tokens[0].start_char == 0
        assert document.sentences[-1].tokens[-1].end_char == 17
        document_text = document.text
        for sentence in document.sentences:
            for token in sentence.tokens:
                assert (
                    document_text[token.start_char : token.end_char]
                    == token.text
                )
            assert isinstance(sentence.sentiment, int)
    assert count == 1
    # One sample where the sample is split into three due to `\n\n` in the
    # middle of the string.
    count = 0
    for document in stanza_batch.batch(
        [EXAMPLE_FOUR],
        nlp,
        clear_cache=clear_cache,
        torch_no_grad=torch_no_grad,
    ):
        count += 1
        # This process removes the `\n \n\n` and adds `\n\n` in its place.
        assert (
            document.text
            == "Hello how are you. Great Thanks\n\nSomething else\n\nAnother test"
        )
        assert document.sentences[0].tokens[0].start_char == 0
        assert document.sentences[-1].tokens[-1].end_char == 61
        document_text = document.text
        for sentence in document.sentences:
            for token in sentence.tokens:
                assert (
                    document_text[token.start_char : token.end_char]
                    == token.text
                )
            assert isinstance(sentence.sentiment, int)
    assert count == 1
    # Multiple samples
    text_dict = {
        0: EXAMPLE_ONE.strip(),
        1: EXAMPLE_THREE.strip(),
        2: "Hello how are you. Great Thanks\n\nSomething else\n\nAnother test",
        3: EXAMPLE_ONE.strip(),
    }
    documents = [EXAMPLE_ONE, EXAMPLE_THREE, EXAMPLE_FOUR, EXAMPLE_ONE]
    count = 0
    for index, document in enumerate(
        stanza_batch.batch(
            documents,
            nlp,
            batch_size=2,
            clear_cache=clear_cache,
            torch_no_grad=torch_no_grad,
        )
    ):
        count += 1
        document_text = document.text
        assert document_text == text_dict[index]
        for sentence in document.sentences:
            for token in sentence.tokens:
                assert (
                    document_text[token.start_char : token.end_char]
                    == token.text
                )
            assert isinstance(sentence.sentiment, int)
    assert count == len(documents)
    # One text across 3 batches
    long_text = "\nHi\n\nNice to meet you\n   \n \nIt is a nice day\n\nBut it could be warmer\n    \nBye!\n\n \n\n"
    count = 0
    for index, document in enumerate(
        stanza_batch.batch(
            [long_text],
            nlp,
            batch_size=2,
            clear_cache=clear_cache,
            torch_no_grad=torch_no_grad,
        )
    ):
        count += 1
        document_text = document.text
        assert (
            document_text
            == "Hi\n\nNice to meet you\n\nIt is a nice day\n\nBut it could be warmer\n\nBye!"
        )
        for sentence in document.sentences:
            for token in sentence.tokens:
                assert (
                    document_text[token.start_char : token.end_char]
                    == token.text
                )
            assert isinstance(sentence.sentiment, int)
    assert count == 1

    # Real world type of test across a number of samples from the Jane Austin
    # book Emma.
    book_data: List[str] = []
    test_data_dir = Path(__file__, "..", "data").resolve()
    with Path(test_data_dir, "jane_austin_emma_data.txt").open(
        "r"
    ) as emma_file:
        book_data = [line for line in emma_file]
    assert len(book_data) == 490

    processed_book_data: List[Document] = []
    processed_book_data = [
        document
        for document in stanza_batch.batch(
            book_data, nlp, clear_cache=clear_cache, torch_no_grad=torch_no_grad
        )
    ]
    assert len(book_data) == len(processed_book_data)
    for true_data, processed_data in zip(book_data, processed_book_data):
        processed_text = processed_data.text
        assert true_data.strip() == processed_text

        for sentence in processed_data.sentences:
            for token in sentence.tokens:
                assert (
                    processed_text[token.start_char : token.end_char]
                    == token.text
                )
            assert isinstance(sentence.sentiment, int)
