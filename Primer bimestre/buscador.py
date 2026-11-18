import csv
import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from io import BytesIO, TextIOWrapper
from urllib.request import urlopen
from zipfile import ZipFile

from snowballstemmer import stemmer


CORPUS = os.getenv("BEIR_CORPUS", "nfcorpus")
MODELS = ("tfidf", "bm25")
STOPWORDS = frozenset("""
    a an and are as at be been being but by can could did do does for from
    had has have he her hers him his how i if in into is it its may me might
    more most my of on or our ours she should so than that the their theirs
    them then there these they this those through to under up us was we were
    what when where which who will with would you your yours
""".split())
TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


@lru_cache(maxsize=65536)
def english_stem(token):
    return stemmer("english").stemWord(token)


def tokenize(text):
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return [
        english_stem(token)
        for token in TOKEN_PATTERN.findall(normalized)
        if token not in STOPWORDS
    ]


@lru_cache(maxsize=2)
def load_beir(corpus):
    if corpus not in ("scifact", "nfcorpus"):
        raise ValueError("BEIR_CORPUS debe ser scifact o nfcorpus")
    base_url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
    with urlopen(f"{base_url}/{corpus}.zip", timeout=90) as response:
        archive = BytesIO(response.read())
    with ZipFile(archive) as data:
        with data.open(f"{corpus}/corpus.jsonl") as source:
            documents = {str(row["_id"]): row for row in map(json.loads, source)}
        with data.open(f"{corpus}/queries.jsonl") as source:
            queries = {str(row["_id"]): row["text"] for row in map(json.loads, source)}
        qrels = defaultdict(dict)
        with TextIOWrapper(data.open(f"{corpus}/qrels/test.tsv"), encoding="utf-8") as source:
            for row in csv.DictReader(source, delimiter="\t"):
                qrels[row["query-id"]][row["corpus-id"]] = int(row["score"])
    return documents, queries, dict(qrels)


class SearchIndex:
    def __init__(self, documents):
        self.documents = documents
        self.postings = defaultdict(dict)
        self.lengths = {}
        for document_id, doc in documents.items():
            counts = Counter(tokenize(f"{doc.get('title') or ''} {doc['text']}"))
            self.lengths[document_id] = counts.total()
            for term, frequency in counts.items():
                self.postings[term][document_id] = frequency

        self.vocabulary = frozenset(self.postings)
        total = len(documents)
        self.average_length = sum(self.lengths.values()) / total
        self.tfidf_idf = {}
        self.bm25_idf = {}
        squared_norms = defaultdict(float)
        for term, posting in self.postings.items():
            df = len(posting)
            idf = math.log((total + 1) / (df + 1)) + 1
            self.tfidf_idf[term] = idf
            self.bm25_idf[term] = math.log1p((total - df + 0.5) / (df + 0.5))
            for document_id, frequency in posting.items():
                squared_norms[document_id] += ((1 + math.log(frequency)) * idf) ** 2
        self.norms = {key: math.sqrt(value) for key, value in squared_norms.items()}

    def rank(self, query, model, top_k):
        terms = Counter(tokenize(query))
        scores = defaultdict(float)
        if model == "tfidf":
            weights = {
                term: (1 + math.log(frequency)) * self.tfidf_idf[term]
                for term, frequency in terms.items()
                if term in self.vocabulary
            }
            query_norm = math.sqrt(sum(weight ** 2 for weight in weights.values()))
            for term, weight in weights.items():
                for doc_id, tf in self.postings[term].items():
                    doc_weight = (1 + math.log(tf)) * self.tfidf_idf[term]
                    scores[doc_id] += weight * doc_weight
            for doc_id in scores:
                scores[doc_id] /= self.norms[doc_id] * query_norm
        elif model == "bm25":
            k1, b = 1.2, 0.75
            for term in terms:
                if term not in self.vocabulary:
                    continue
                for doc_id, tf in self.postings[term].items():
                    length_ratio = self.lengths[doc_id] / self.average_length
                    denominator = tf + k1 * (1 - b + b * length_ratio)
                    scores[doc_id] += self.bm25_idf[term] * tf * (k1 + 1) / denominator
        else:
            raise ValueError("Modelo desconocido")
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]

    def search(self, query, model, top_k):
        results = []
        for rank, (document_id, score) in enumerate(self.rank(query, model, top_k), 1):
            doc = self.documents[document_id]
            text = " ".join(doc["text"].split())
            results.append({
                "rank": rank,
                "document_id": document_id,
                "score": score,
                "title": doc.get("title") or "",
                "snippet": text[:240] + ("…" if len(text) > 240 else ""),
            })
        return {"model": model, "results": results}


def create_app(index, corpus):
    def application(environ, start_response):
        def respond(status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            start_response(status, [
                ("Content-Type", "application/json; charset=utf-8"),
                ("Content-Length", str(len(body))),
            ])
            return [body]

        path = environ["PATH_INFO"]
        method = environ["REQUEST_METHOD"]
        if path == "/health" and method == "GET":
            return respond("200 OK", {
                "status": "ok", "corpus": corpus, "documents": len(index.documents),
            })
        if path != "/search" or method != "POST":
            return respond("404 Not Found", {"error": "Ruta o método desconocido"})
        try:
            length = int(environ.get("CONTENT_LENGTH") or 0)
            if length > 65536:
                return respond("413 Payload Too Large", {"error": "Solicitud demasiado grande"})
            if length <= 0:
                raise ValueError
            request = json.loads(environ["wsgi.input"].read(length))
        except (ValueError, RecursionError):
            return respond("400 Bad Request", {"error": "Envía un objeto JSON válido"})
        if not isinstance(request, dict):
            return respond("400 Bad Request", {"error": "Envía un objeto JSON"})
        query = request.get("query")
        model = request.get("model")
        top_k = request.get("top_k", 10)
        if not isinstance(query, str) or not query.strip():
            return respond("400 Bad Request", {"error": "query debe ser texto no vacío"})
        if model not in MODELS:
            return respond("400 Bad Request", {"error": "model debe ser bm25 o tfidf"})
        if type(top_k) is not int or top_k < 1:
            return respond("400 Bad Request", {"error": "top_k debe ser un entero positivo"})
        return respond("200 OK", index.search(query, model, min(top_k, len(index.documents))))

    return application


def production_app():
    documents, _, _ = load_beir(CORPUS)
    return create_app(SearchIndex(documents), CORPUS)


if __name__ == "__main__":
    from wsgiref.simple_server import make_server

    with make_server("127.0.0.1", 8000, production_app()) as server:
        print("http://127.0.0.1:8000/health", flush=True)
        server.serve_forever()
