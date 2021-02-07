from typing import Iterable, Tuple, List, Dict, Pattern
import re

import stanza
from stanza.models.common.doc import Document, Sentence

# If Stanza version 1.1
if stanza.__version__ == "1.1.1":
    from stanza.models.tokenize.data import NEWLINE_WHITESPACE_RE
else:
    from stanza.models.tokenization.data import NEWLINE_WHITESPACE_RE
import torch

from stanza_batch.version import VERSION as __version__

# NEWLINE_WHITESPACE_RE = re.compile(r'\n\s*\n') That is what NEWLINE_WHITESPACE_RE
# is within Stanza
START_OFFSET_RE = re.compile(r"start_char=(\d+)")
END_OFFSET_RE = re.compile(r"end_char=(\d+)")


def _start_end_character_offsets(token_misc: str) -> Tuple[int, int]:
    """
    :param token_misc: The `misc` key's value from a Stanza token.
    :returns: The start and end character offset for the given token from that
              tokens `misc` key's value.
    """

    def _offset(_misc: str, regex: Pattern) -> int:
        _match = regex.search(_misc)
        assert _match is not None
        offset_str = _match.groups()[0]
        assert offset_str is not None
        return int(offset_str)

    return (
        _offset(token_misc, START_OFFSET_RE),
        _offset(token_misc, END_OFFSET_RE),
    )


def _stanza_batch(
    data: Iterable[str], batch_size: int = 32
) -> Iterable[Tuple[str, List[int], List[int]]]:
    """
    Batches text together so that Stanza can process them quicker, rather than
    giving Stanza one text at a time to process. The way this batches text
    together is by joining the texts together with `\n\n` as suggested in the
    Stanza documentation:
    https://stanfordnlp.github.io/stanza/pipeline.html#usage

    However it will split a given document into smaller documents within the
    batch using the following regular expression: re.compile('\\n\\s*\\n')
    Thus if your single document is `hello\n \nhow are you` this will be
    processed as two separate paragraphs e.g. `hello` and `how are you`. The
    list of document indexes that are produced as output will allow you to
    know if one of your documents has been split into two or more pieces e.g.
    in the last case the returned document indexes will be [0,0] as the two
    separate documents have come from the same one document that was the input.

    :param data: A list/iterable of texts you want tagging.
    :param batch_size: The number of texts to process at one time.
    :returns: The Tuple of length 3 where the first items is the batched up
              texts, the second are the character offsets that denoted the end
              of a text/paragraph within the batch and the last the list of
              document indexes.
    :raises ValueError: If a sample in the data contains no text after being
                        split using `re.compile('\\n\\s*\\n')` regular expression.
    """
    batch_str = ""
    document_offsets: List[int] = []
    document_indexes: List[int] = []
    current_end_offset = 0
    current_batch_size = 0
    document_index = 0
    for sample in data:
        sub_samples = NEWLINE_WHITESPACE_RE.split(sample)
        if not "".join(sub_samples).strip():
            raise ValueError("A sample contains no text.")
        for sub_sample in sub_samples:
            batch_str += f"{sub_sample}"
            current_batch_size += 1
            if current_batch_size == batch_size:
                current_end_offset += len(sub_sample)
                batch_str += "\n\n"
                document_offsets.append(current_end_offset)
                document_indexes.append(document_index)
                yield batch_str, document_offsets, document_indexes
                batch_str = ""
                current_end_offset = 0
                document_offsets = []
                current_batch_size = 0
                document_indexes = []
            else:
                current_end_offset += len(sub_sample)
                document_offsets.append(current_end_offset)
                batch_str += "\n\n"
                current_end_offset += len("\n\n")
                document_indexes.append(document_index)
        document_index += 1
    if batch_str:
        yield batch_str, document_offsets, document_indexes


def _create_stanza_document(
    sentence_dicts: List[List[Dict[str, str]]], document_text: str
) -> Document:
    stanza_document = Document(sentence_dicts, text=document_text)
    contains_entities = False
    for sentence_index, sentence_dict in enumerate(sentence_dicts):
        first_token = sentence_dict[0]
        sentence_sentiment = first_token["sentence_sentiment"]
        if sentence_sentiment is not None:
            stanza_document.sentences[
                sentence_index
            ].sentiment = sentence_sentiment
        if "ner" in first_token:
            contains_entities = True
    if contains_entities:
        stanza_document.build_ents()
    return stanza_document


def _batch_to_documents(
    processed_batch: Document,
    document_offsets: List[int],
    document_indexes: List[int],
) -> Tuple[List[Document], List[int]]:
    """
    Given a Stanza Document that was the output of a Stanza model where the
    input was a batch String from `_stanza_batch` method, it returns a list
    of Stanza Documents one for each data item in the original batch.

    :param processed_batch: Stanza Document that was the output of a Stanza
                            model where the input was a batch String from
                            `_stanza_batch` method
    :param document_offsets: The `character offsets` from the output of
                             `_stanza_batch`.
    :param document_indexes: The `document indexes` from the output of
                             `_stanza_batch`.
    :returns: A list of Stanza Documents one for each data item in the
              original batch and a list of their original document indexes.
    """

    def get_start_end_offset(
        relevant_sentences: List[Sentence],
    ) -> Tuple[int, int]:
        start_offset = relevant_sentences[0].tokens[0].start_char
        end_offset = relevant_sentences[-1].tokens[-1].end_char
        return start_offset, end_offset

    def change_offsets(
        relevant_sentences: List[Sentence], start_offset: int
    ) -> List[List[Dict[str, str]]]:
        assert start_offset != -1
        all_sentence_dicts: List[List[Dict[str, str]]] = []
        for sentence in relevant_sentences:
            sentence_dicts: List[Dict[str, str]] = []
            sentence_sentiment = getattr(sentence, "sentiment", None)
            for token in sentence.to_dict():
                token_misc = token["misc"]
                start, end = _start_end_character_offsets(token_misc)
                start = start - start_offset
                end = end - start_offset
                token["misc"] = f"start_char={start}|end_char={end}"
                token["sentence_sentiment"] = sentence_sentiment
                sentence_dicts.append(token)
            all_sentence_dicts.append(sentence_dicts)
        return all_sentence_dicts

    batch_documents: List[Document] = []
    batch_document_indexes: List[int] = []

    previous_document_sentences: List[Sentence] = []
    document_sentences: List[Sentence] = []
    current_document_index = 0
    current_document = document_indexes[current_document_index]
    current_offset_index = 0
    current_offset = document_offsets[current_offset_index]

    original_text = processed_batch.text
    document_text = ""
    for sentence in processed_batch.sentences:
        last_token = sentence.tokens[-1]
        offset = last_token.end_char
        if offset <= current_offset:
            document_sentences.append(sentence)
        else:
            current_offset_index += 1
            current_offset = document_offsets[current_offset_index]

            # Checking to see if the next document is part of the same document
            current_document_index += 1
            if current_document == document_indexes[current_document_index]:
                document_sentences.append(sentence)
                start, end = get_start_end_offset(document_sentences)
                if document_text:
                    document_text += "\n\n" + original_text[start:end]
                else:
                    document_text = original_text[start:end]
                previous_document_sentences.extend(document_sentences)
                document_sentences = []
            else:
                start = -1
                if document_sentences:
                    start, end = get_start_end_offset(document_sentences)
                    document_text += original_text[start:end]

                document_sentences.extend(previous_document_sentences)

                if previous_document_sentences:
                    start, _ = get_start_end_offset(previous_document_sentences)

                document_sentence_dicts = change_offsets(
                    document_sentences, start
                )
                document = _create_stanza_document(
                    document_sentence_dicts, document_text
                )

                batch_documents.append(document)
                batch_document_indexes.append(current_document)
                document_text = ""
                document_sentences = [sentence]
                previous_document_sentences = []
            current_document = document_indexes[current_document_index]

    if document_sentences or previous_document_sentences:
        start = 0
        if document_sentences:
            start, end = get_start_end_offset(document_sentences)
            document_text += original_text[start:end]

        if previous_document_sentences:
            start, _ = get_start_end_offset(previous_document_sentences)
        document_sentences.extend(previous_document_sentences)
        document_sentence_dicts = change_offsets(document_sentences, start)
        document = _create_stanza_document(
            document_sentence_dicts, document_text
        )
        batch_documents.append(document)
        batch_document_indexes.append(current_document)
        document_text = ""

    return batch_documents, batch_document_indexes


def combine_stanza_documents(stanza_documents: List[Document]) -> Document:
    """
    :param stanza_documents: The stanza documents to be combined.
    :returns: The stanza documents combined into one document whereby they are
              combined in order whereby the first document will contain the
              first token in the new document and the last document will contain
              the last token in the new document. NOTE: The text of all documents
              are combined so that there are \n\n seperating the combined texts.
    """

    def change_offsets(
        relevant_sentences: List[Sentence], offset_to_add: int
    ) -> List[List[Dict[str, str]]]:
        all_sentence_dicts: List[List[Dict[str, str]]] = []
        for sentence in relevant_sentences:
            sentence_dicts: List[Dict[str, str]] = []
            sentence_sentiment = getattr(sentence, "sentiment", None)
            for token in sentence.to_dict():
                token_misc = token["misc"]
                start, end = _start_end_character_offsets(token_misc)
                start = start + offset_to_add
                end = end + offset_to_add
                token["misc"] = f"start_char={start}|end_char={end}"
                token["sentence_sentiment"] = sentence_sentiment
                sentence_dicts.append(token)
            all_sentence_dicts.append(sentence_dicts)
        return all_sentence_dicts

    character_offset_to_add = 0
    new_sentences: List[List[Dict[str, str]]] = []
    new_document_text = ""
    for index, document in enumerate(stanza_documents):
        new_sentences.extend(
            change_offsets(document.sentences, character_offset_to_add)
        )
        document_text = document.text
        if index == (len(stanza_documents) - 1):
            new_document_text += document_text
        else:
            new_document_text += f"{document_text}\n\n"
        character_offset_to_add = len(new_document_text)

    return _create_stanza_document(new_sentences, new_document_text)


def batch(
    data: Iterable[str],
    stanza_pipeline: stanza.Pipeline,
    batch_size: int = 32,
    clear_cache: bool = True,
    torch_no_grad: bool = True,
) -> Iterable[Document]:
    """
    Batch processes the given texts using the given Stanza pipeline.
    The way this batches text together is by joining the texts together with
    `\n\n` as suggested in the Stanza documentation:
    https://stanfordnlp.github.io/stanza/pipeline.html#usage

    However it will split a given text into smaller texts within the
    batch using the following regular expression: re.compile('\\n\\s*\\n')
    Thus if your text is `hello\n \nhow are you` this will be
    processed as two separate paragraphs e.g. `hello` and `how are you`, but when
    returned it will be in one Stanza document as `hello\n\nhow are you`
    (texts when split using the re.compile('\\n\\s*\\n') expression are joined
    back together with `\n\n`). Further if your text contains any whitespace
    at the begining or end of the text this will be removed.

    :param data: A list/iterable of texts you want processing.
    :param stanza_pipeline: The Stanza pipeline used to process the texts.
    :param batch_size: The number of texts to process at one time.
    :param clear_cache: If True will call the python garbage collector and if
                        using the GPU will empty the CUDA cache after each
                        batch has been processed. This is to stop the memory
                        from being consumed for both the main memory and GPU
                        memory over the whole python process.
    :param torch_no_grad: This wraps the stanza nlp pipeline in a `torch.no_grad`
                          context manager, this reduces the memory used by torch
                          and can speed up the pipeline. This turns off the
                          autograd engine that is used by PyTorch for the
                          stanza nlp pipeline when True.
    :returns: An Iterable of processed texts represented as Stanza Documents.
              This will be of the same length as the data iterable given.
    :raises ValueError: If a sample in the data contains no text after being
                        split using `re.compile('\\n\\s*\\n')` regular expression.
    """
    documents_across_batches: List[Document] = []
    batch_last_document_index = (
        -1
    )  # a batch index can never be -1, just here for initialization
    for batch_str, _offsets, _doc_indexes in _stanza_batch(
        data, batch_size=batch_size
    ):
        if torch_no_grad:
            with torch.no_grad():
                stanza_document = stanza_pipeline(batch_str)
        else:
            stanza_document = stanza_pipeline(batch_str)
        (
            processed_stanza_documents,
            processed_document_indexes,
        ) = _batch_to_documents(stanza_document, _offsets, _doc_indexes)

        # Yield documents that have come from the last *n* batches.
        if batch_last_document_index != processed_document_indexes[-1]:
            new_processed_document_indexes: List[int] = []
            for index, processed_document_index in enumerate(
                processed_document_indexes
            ):
                if batch_last_document_index == processed_document_index:
                    documents_across_batches.append(
                        processed_stanza_documents.pop(index)
                    )
                else:
                    new_processed_document_indexes.append(
                        processed_document_index
                    )
            if documents_across_batches:
                yield combine_stanza_documents(documents_across_batches)
            documents_across_batches = []
            processed_document_indexes = new_processed_document_indexes

        # Yield documents that have come from the current batch
        batch_last_document_index = processed_document_indexes[-1]
        for processed_stanza_document, processed_document_index in zip(
            processed_stanza_documents, processed_document_indexes
        ):
            if processed_document_index == batch_last_document_index:
                documents_across_batches.append(processed_stanza_document)
            else:
                yield processed_stanza_document
        if clear_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
    if documents_across_batches:
        yield combine_stanza_documents(documents_across_batches)
    if clear_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()
