# app.py
import streamlit as st
import re
import math
import torch
import numpy as np
import faiss
from datasketch import MinHash, MinHashLSH
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher
from io import BytesIO
from docx import Document
import torch.nn.functional as F
from collections import Counter

# ---------------------------
# CONFIGURATION
# ---------------------------
SHINGLE_K = 5
MINHASH_PERM = 128
LSH_THRESHOLD = 0.3
EMBED_MODEL_NAME = "paraphrase-MiniLM-L12-v2"
GPT2_MODEL = "distilgpt2"
USE_FAISS = True

# ---------------------------
# TEXT UTILITIES
# ---------------------------
def normalize_text(t: str):
    return re.sub(r'\s+', ' ', t).strip()

def simple_tokenize(text: str):
    """Offline tokenizer: splits words by non-alphanumeric characters."""
    return re.findall(r'\w+', text.lower())

def get_word_shingles(text: str, k=SHINGLE_K):
    tokens = simple_tokenize(text)
    return {" ".join(tokens[i:i+k]) for i in range(len(tokens) - k + 1)}

def minhash_from_shingles(shingles):
    m = MinHash(num_perm=MINHASH_PERM)
    for s in shingles:
        m.update(s.encode('utf8'))
    return m

def jaccard(set_a, set_b):
    return len(set_a & set_b) / len(set_a | set_b) if set_a or set_b else 0.0

def longest_matching_spans(a: str, b: str, min_len=20):
    s = SequenceMatcher(None, a, b)
    return [(a[i:i+l], i, j, l) for i, j, l in s.get_matching_blocks() if l >= min_len]

def extract_text_from_docx(file_bytes: BytesIO) -> str:
    """Extract text from uploaded .docx Word file."""
    try:
        doc = Document(file_bytes)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        st.error(f"Failed to read Word file: {e}")
        return ""

# ---------------------------
# PLAGIARISM INDEX
# ---------------------------
class PlagiarismIndex:
    def __init__(self):
        self.doc_texts = {}
        self.doc_shingles = {}
        self.doc_minhashes = {}
        self.lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=MINHASH_PERM)
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        self.corpus_embeddings = None
        self.corpus_ids = []
        self.faiss_index = None

    def add_document(self, doc_id, text):
        text = normalize_text(text)
        if not text:
            raise ValueError("Empty document text.")
        self.doc_texts[doc_id] = text
        shingles = get_word_shingles(text)
        self.doc_shingles[doc_id] = shingles
        m = minhash_from_shingles(shingles)
        self.doc_minhashes[doc_id] = m
        self.lsh.insert(doc_id, m)

        emb = self.embed_model.encode([text], convert_to_numpy=True)
        if self.corpus_embeddings is None:
            self.corpus_embeddings = emb
        else:
            self.corpus_embeddings = np.vstack([self.corpus_embeddings, emb])
        self.corpus_ids.append(doc_id)

        if USE_FAISS:
            self._build_faiss()

    def _build_faiss(self):
        emb = self.corpus_embeddings.astype('float32')
        faiss.normalize_L2(emb)
        self.faiss_index = faiss.IndexFlatIP(emb.shape[1])
        self.faiss_index.add(emb)

    def query(self, text, top_k=5):
        text = normalize_text(text)
        shingles = get_word_shingles(text)
        q_m = minhash_from_shingles(shingles)
        candidates = self.lsh.query(q_m)

        emb = self.embed_model.encode([text], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(emb)
        emb_results = []

        if self.faiss_index is not None:
            D, I = self.faiss_index.search(emb, top_k)
            for score, idx in zip(D[0], I[0]):
                if idx < len(self.corpus_ids) and score > 0:
                    emb_results.append(self.corpus_ids[idx])
        else:
            sims = []
            qn = emb[0] / np.linalg.norm(emb[0])
            for i, cid in enumerate(self.corpus_ids):
                v = self.corpus_embeddings[i]
                sims.append((np.dot(qn, v / np.linalg.norm(v)), cid))
            sims = sorted(sims, reverse=True)[:top_k]
            emb_results.extend([cid for _, cid in sims])

        final_ids = list(dict.fromkeys(candidates + emb_results))
        results = []
        for cid in final_ids:
            jac = jaccard(shingles, self.doc_shingles[cid])
            results.append({
                "doc_id": cid,
                "jaccard": jac,
                "text": self.doc_texts[cid]
            })
        return sorted(results, key=lambda x: x["jaccard"], reverse=True)

# ---------------------------
# AI-LIKENESS DETECTOR
# ---------------------------
class AIDetector:
    def __init__(self, model_name=GPT2_MODEL, device=None):
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.n_positions = getattr(self.model.config, "n_positions", 1024)

    def _per_token_logprobs(self, enc_ids, chunk_size=512, overlap=64):
        n = enc_ids.size(0)
        if n <= 1: return []
        token_logps = [None]*n
        stride = max(1, chunk_size - overlap)
        for start in range(0, n, stride):
            end = min(start + chunk_size, n)
            chunk = enc_ids[start:end].unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(chunk).logits
                log_probs = F.log_softmax(logits, dim=-1)
            L = chunk.size(1)
            if L <= 1: continue
            target_ids = chunk[0, 1:]
            lp = log_probs[0, :-1, :]
            tok_lp_values = lp[torch.arange(L-1), target_ids].cpu().tolist()
            for i, val in enumerate(tok_lp_values):
                global_idx = start + i + 1
                if global_idx < n and token_logps[global_idx] is None:
                    token_logps[global_idx] = val
            if all(x is not None for x in token_logps[1:]): break
        filled = [lp for lp in token_logps[1:] if lp is not None]
        return filled

    def perplexity(self, text, chunk_size=None, overlap=64):
        if not text.strip(): return float("inf")
        enc = self.tokenizer.encode(text, return_tensors="pt")[0]
        if chunk_size is None: chunk_size = min(self.n_positions, 512)
        per_token_logps = self._per_token_logprobs(enc, chunk_size=chunk_size, overlap=overlap)
        if not per_token_logps: return float("inf")
        avg_logp = sum(per_token_logps)/len(per_token_logps)
        ppl = math.exp(-avg_logp)
        return float(ppl)

    def _repetition_score(self, text):
        toks = [t.lower() for t in self.tokenizer.tokenize(text) if t.strip()]
        if not toks: return 0.0
        counts = Counter(toks)
        max_freq = max(counts.values())
        fraction = max_freq/len(toks)
        return min(100.0, fraction*100*2)

    def _ttr_score(self, text):
        toks = [t.lower() for t in self.tokenizer.tokenize(text) if t.strip()]
        if not toks: return 0.0
        ttr = len(set(toks))/len(toks)
        return (1 - ttr)*100

    def ai_score(self, text):
        ppl = self.perplexity(text)
        if math.isinf(ppl): return {"ppl": None, "score": None}
        logppl = math.log(max(ppl,1e-8))
        low, high = math.log(10), math.log(200)
        frac = max(0.0, min(1.0, (logppl-low)/(high-low)))
        perplexity_score = frac*100
        rep = self._repetition_score(text)
        ttr_s = self._ttr_score(text)
        w_ppl, w_rep, w_ttr = 0.6, 0.25, 0.15
        final = perplexity_score*w_ppl + rep*w_rep + ttr_s*w_ttr
        final = max(0.0, min(100.0, final))
        return {"ppl": round(ppl,2), "score": round(final,2),
                "components": {"perplexity": round(perplexity_score,2),
                               "repetition": round(rep,2),
                               "ttr": round(ttr_s,2)}}
