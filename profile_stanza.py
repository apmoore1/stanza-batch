import argparse
from pathlib import Path
from typing import List
from time import time
from pathlib import Path

import stanza
from stanza.models.common.doc import Document
import stanza_batch
import GPUtil
import matplotlib.pyplot as plt

def path_type(fp: str) -> Path:
    return Path(fp).resolve()

if __name__ == "__main__":
    save_fp_help = ('File path to save the GPU memory as a '
                    'function of documents processed plot. '
                    'As we have to get the GPU memory usage for each '
                    'document this slows the program down a lot.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-fp', help=save_fp_help, type=path_type)
    parser.add_argument('--clear-cache', action='store_true')
    args = parser.parse_args()

    stanza.download("en", processors="tokenize,pos,sentiment")
    nlp = stanza.Pipeline(
        lang="en", processors="tokenize,pos,sentiment", use_gpu=True
    )

    book_data: List[str] = []
    test_data_dir = Path(__file__, "..", "tests", "data").resolve()
    with Path(test_data_dir, "jane_austin_emma_data.txt").open(
        "r"
    ) as emma_file:
        book_data = [line for line in emma_file]
    assert len(book_data) == 490

    t = time()
    gpu_memory_used: List[float] = []
    processed_book_data: List[Document] = []
    for document in stanza_batch.batch(book_data, nlp, clear_cache=args.clear_cache):
        processed_book_data.append(document)
        if args.save_fp:
            # assuming the first GPU is the one being used.
            gpu_memory_used.append(GPUtil.getGPUs()[0].memoryUsed)
    print(f'Time taken: {time() - t}')

    if args.save_fp:
        number_documents_processed = range(len(processed_book_data))
        plt.plot(number_documents_processed, gpu_memory_used)
        plt.xlabel('Number of documents processed')
        plt.ylabel('GPU Memory used (MB)')
        plt.grid(True)
        plt.savefig(str(args.save_fp))
