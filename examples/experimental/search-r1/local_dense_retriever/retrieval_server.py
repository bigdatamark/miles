# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/search/retrieval_server.py

# [MODULE DOC] Local retrieval web server for the Search-R1 example. Provides retrievers (a BM25
# one using a Lucene index, and a dense one using a FAISS index plus a transformer encoder on GPU)
# behind a common base, a corpus loader, and a single POST /retrieve endpoint that accepts a batch
# of query strings and returns the top documents for each (optionally with scores). When run as a
# script it parses the index path, corpus path, top-k (3), retriever name (e5) and model path
# (intfloat/e5-base-v2) and GPU-index flag, builds one global retriever (half precision, query
# length 256, batch size 512, mean pooling) and serves on all interfaces at port 8000.
import argparse
import json
import warnings

import datasets
import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def load_corpus(corpus_path: str):
    # [FUNCTION DOC] HELPER: Loads a JSON-lines corpus file as the train split of a Hugging Face
    # dataset using 4 processes and returns the dataset (indexable by integer position).
    corpus = datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)
    return corpus


def load_docs(corpus, doc_idxs):
    # [FUNCTION DOC] HELPER: Returns the list of corpus records at the given positions (each
    # converted to an integer), in order, one lookup at a time.
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results


def load_model(model_path: str, use_fp16: bool = False):
    # [FUNCTION DOC] HELPER: Loads a transformer encoder (remote code trusted) in evaluation mode
    # onto the current CUDA device, converting to half precision if requested, and loads the fast
    # tokenizer. Returns the pair of model and tokenizer. Requires a GPU.
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer


def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    # [FUNCTION DOC] HELPER: Reduces token embeddings (batch, length, hidden) to sentence embeddings
    # (batch, hidden). Mean: zeroes padded positions using the attention mask (required) and divides
    # the sum over tokens by the number of real tokens. Cls: the first token's vector. Pooler: the
    # model's pooled output as is. Any other method raises a not-implemented error.
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")


class Encoder:
    # [CLASS DOC] Text encoder wrapping a transformer on GPU. Stores the model name (used to choose
    # prefix and normalization rules), path, pooling method, maximum token length and half
    # precision flag, and holds the loaded model and tokenizer.
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        # [FUNCTION DOC] Stores the settings, loads the model and tokenizer (GPU required) and puts
        # the model in evaluation mode.
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: list[str], is_query=True) -> np.ndarray:
        # [FUNCTION DOC] Encodes texts to a float32 C-contiguous array of shape (count, dimension)
        # without gradients. A single string is wrapped into a list. Prefixes by model name (case-
        # insensitive substring): names containing e5 get "query: " (queries) or "passage: "
        # (documents); names containing bge get the retrieval instruction prefix for queries only.
        # Tokenizes with padding and truncation to the maximum length and moves inputs to the GPU.
        # T5-type models: runs with a single zero decoder start token and takes the first decoder
        # hidden state. Others: pools with the configured method and, unless the model name contains
        # dpr, L2-normalizes. Copies to CPU, frees GPU tensors and empties the CUDA cache.
        # processing query for different encoders
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [
                    f"Represent this sentence for searching relevant passages: {query}" for query in query_list
                ]

        inputs = self.tokenizer(
            query_list, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long).to(
                inputs["input_ids"].device
            )
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(
                output.pooler_output, output.last_hidden_state, inputs["attention_mask"], self.pooling_method
            )
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")

        del inputs, output
        torch.cuda.empty_cache()

        return query_emb


class BaseRetriever:
    # [CLASS DOC] Abstract retriever. Holds the config and copies the retrieval method name,
    # default top-k, index path and corpus path from it. Subclasses implement the single-query and
    # batch search methods; the public methods delegate to them.
    def __init__(self, config):
        # [FUNCTION DOC] Stores the config and the fields listed in the class description.
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk

        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool):
        # [FUNCTION DOC] HELPER: Abstract single-query search; raises a not-implemented error.
        raise NotImplementedError

    def _batch_search(self, query_list: list[str], num: int, return_score: bool):
        # [FUNCTION DOC] HELPER: Abstract batch search; raises a not-implemented error.
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        # [FUNCTION DOC] Searches one query by delegating to the subclass; returns the documents
        # (and the scores as a second item when return_score is true). num none means default top-k.
        return self._search(query, num, return_score)

    def batch_search(self, query_list: list[str], num: int = None, return_score: bool = False):
        # [FUNCTION DOC] Searches many queries by delegating to the subclass; returns a list of
        # per-query document lists (and a list of per-query score lists as a second item when
        # return_score is true).
        return self._batch_search(query_list, num, return_score)


class BM25Retriever(BaseRetriever):
    # [CLASS DOC] Sparse retriever over a Lucene BM25 index (pyserini imported lazily). Documents
    # come from the index itself when it stores raw documents, otherwise from the separately loaded
    # corpus dataset looked up by document id.
    def __init__(self, config):
        # [FUNCTION DOC] Opens the Lucene index at the index path, checks whether the index stores
        # raw documents (by reading document zero), loads the corpus dataset only if it does not,
        # and sets a process-count attribute to 8.
        # KNOWN ISSUE: the process-count attribute is never used.
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher

        self.searcher = LuceneSearcher(self.index_path)
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)
        self.max_process_num = 8

    def _check_contain_doc(self):
        # [FUNCTION DOC] HELPER: True if the raw stored document at id zero exists in the index.
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score: bool = False):
        # [FUNCTION DOC] HELPER: Runs one BM25 query for num hits (default top-k). No hits: returns
        # empty documents (and an empty score list when scores are requested). Scores are taken from
        # all hits; if fewer than num hits came back a warning is emitted, else hits are cut to num.
        # Documents: when the index stores raw documents, each is parsed from its stored JSON and
        # converted to a mapping with title (first content line stripped of double quotes), text
        # (remaining lines) and contents; otherwise records are loaded from the corpus by document
        # id. Returns the documents, plus the scores when requested.
        if num is None:
            num = self.topk
        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            if return_score:
                return [], []
            else:
                return []
        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn("Not enough documents retrieved!", stacklevel=2)
        else:
            hits = hits[:num]

        if self.contain_doc:
            all_contents = [json.loads(self.searcher.doc(hit.docid).raw())["contents"] for hit in hits]
            results = [
                {
                    "title": content.split("\n")[0].strip('"'),
                    "text": "\n".join(content.split("\n")[1:]),
                    "contents": content,
                }
                for content in all_contents
            ]
        else:
            results = load_docs(self.corpus, [hit.docid for hit in hits])

        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: list[str], num: int = None, return_score: bool = False):
        # [FUNCTION DOC] HELPER: Runs the single-query search sequentially for each query (always
        # requesting scores) and returns the list of document lists, plus the list of score lists
        # when requested.
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num, True)
            results.append(item_result)
            scores.append(item_score)
        if return_score:
            return results, scores
        else:
            return results


class DenseRetriever(BaseRetriever):
    # [CLASS DOC] Dense retriever: a FAISS index of passage embeddings, the corpus dataset, and an
    # encoder for queries. Scores are the index's similarity values.
    def __init__(self, config):
        # [FUNCTION DOC] Reads the FAISS index from disk; if the GPU option is set, copies it to all
        # GPUs sharded with half-precision storage. Loads the corpus, builds the encoder from the
        # config (method name as model name, model path, pooling method, query max length, half
        # precision flag), and takes top-k and batch size from the config.
        super().__init__(config)
        self.index = faiss.read_index(self.index_path)
        if config.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

        self.corpus = load_corpus(self.corpus_path)
        self.encoder = Encoder(
            model_name=self.retrieval_method,
            model_path=config.retrieval_model_path,
            pooling_method=config.retrieval_pooling_method,
            max_length=config.retrieval_query_max_length,
            use_fp16=config.retrieval_use_fp16,
        )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def _search(self, query: str, num: int = None, return_score: bool = False):
        # [FUNCTION DOC] HELPER: Encodes the query, searches the index for num nearest passages
        # (default top-k), loads the corresponding corpus records and returns them, plus the scores
        # as a plain list when requested.
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]
        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores.tolist()
        else:
            return results

    def _batch_search(self, query_list: list[str], num: int = None, return_score: bool = False):
        # [FUNCTION DOC] HELPER: Batched dense search. A single string becomes a one-item list; num
        # defaults to top-k. Processes queries in chunks of the configured batch size (with a
        # progress bar): encode, search the index, flatten the index ids, load all records, regroup
        # them per query into groups of num, and release GPU memory. Returns the list of per-query
        # record lists, plus the per-query score lists when requested.
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk

        results = []
        scores = []
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc="Retrieval process: "):
            query_batch = query_list[start_idx : start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            # load_docs is not vectorized, but is a python list approach
            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            # chunk them back
            batch_results = [batch_results[i * num : (i + 1) * num] for i in range(len(batch_idxs))]

            results.extend(batch_results)
            scores.extend(batch_scores)

            del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results
            torch.cuda.empty_cache()

        if return_score:
            return results, scores
        else:
            return results


def get_retriever(config):
    # [FUNCTION DOC] Factory: returns a BM25 retriever when the config's retrieval method is
    # "bm25", otherwise a dense retriever. Loads indexes and models (heavy side effects).
    if config.retrieval_method == "bm25":
        return BM25Retriever(config)
    else:
        return DenseRetriever(config)


#####################################
# FastAPI server below
#####################################


class Config:
    # [CLASS DOC] HELPER: Plain settings holder standing in for parsed arguments. Attributes (with
    # defaults): retrieval method (bm25), top-k (10), index path, corpus path, dataset path, data
    # split (train), faiss-on-GPU flag (true), encoder model path, pooling method (mean), query
    # maximum length (256), half precision flag (false), batch size (128).
    # KNOWN ISSUE: the dataset path and data split attributes are never read.
    """
    Minimal config class (simulating your argparse)
    Replace this with your real arguments or load them dynamically.
    """

    def __init__(
        self,
        retrieval_method: str = "bm25",
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        faiss_gpu: bool = True,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128,
    ):
        # [FUNCTION DOC] HELPER: Stores each argument as an attribute of the same name.
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size


class QueryRequest(BaseModel):
    # [CLASS DOC] Request body of the retrieve endpoint: a list of query strings, an optional top-k
    # (none or zero falls back to the server default), and a flag to return scores (default false).
    queries: list[str]
    topk: int | None = None
    return_scores: bool = False


app = FastAPI()


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    # [FUNCTION DOC] POST /retrieve handler. Uses the module-level config and retriever globals
    # that exist only when the file runs as a script. If top-k is unset or zero, sets it on the
    # request to the config default. Runs batch retrieval for all queries (with scores if
    # requested), then builds the response: a mapping with key "result" holding, per query, either
    # a list of {"document": record, "score": value} items (when scores were produced) or the
    # plain record list.
    # KNOWN ISSUE: results are separated from scores by trying to unpack the return value into two
    # parts; when scores are not requested and exactly two queries are sent, the unpacking succeeds
    # by accident and the second query's records are treated as scores, producing wrong or failing
    # output. Also the docstring shows a doubled nested mapping for scored items that does not
    # match the actual output.
    """
    Endpoint that accepts queries and performs retrieval.

    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
    }

    Output format (when return_scores=True，similarity scores are returned):
    {
        "result": [
            [   # Results for each query
                {
                    {"document": doc, "score": score}
                },
                # ... more documents
            ],
            # ... results for other queries
        ]
    }
    """
    if not request.topk:
        request.topk = config.retrieval_topk  # fallback to default

    # Perform batch retrieval
    tmp = retriever.batch_search(query_list=request.queries, num=request.topk, return_score=request.return_scores)

    scores = []
    try:
        results, scores = tmp
    except ValueError:
        results = tmp

    # Format response
    resp = []
    for i, single_result in enumerate(results):
        if scores:
            # If scores are returned, combine them with results
            combined = []
            for doc, score in zip(single_result, scores[i], strict=True):
                combined.append({"document": doc, "score": score})
            resp.append(combined)
        else:
            resp.append(single_result)
    return {"result": resp}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
    parser.add_argument(
        "--index_path",
        type=str,
        default="/home/peterjin/mnt/index/wiki-18/e5_Flat.index",
        help="Corpus indexing file.",
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl",
        help="Local corpus file.",
    )
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model.")
    parser.add_argument(
        "--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the retriever model."
    )
    parser.add_argument("--faiss_gpu", action="store_true", help="Use GPU for computation")

    args = parser.parse_args()

    # 1) Build a config (could also parse from arguments).
    #    In real usage, you'd parse your CLI arguments or environment variables.
    config = Config(
        retrieval_method=args.retriever_name,  # or "dense"
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512,
    )

    # 2) Instantiate a global retriever so it is loaded once and reused.
    retriever = get_retriever(config)

    # 3) Launch the server. By default, it listens on http://127.0.0.1:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
