import os
import json
import hashlib
import requests
from functools import wraps
from flask import (
    Flask, render_template, request, jsonify,
    session, redirect, url_for
)
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY")

# ──────────────────────────────────────────────
# CONFIG  (dari environment variable atau .env)
# ──────────────────────────────────────────────
APP_PASSWORD = os.environ.get("APP_PASSWORD")
APP_USERNAME = os.environ.get("APP_USERNAME")

QDRANT_URL        = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY    = os.environ.get("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION")

NIM_URL    = os.environ.get("NIM_URL")
NIM_API_KEY = os.environ.get("NIM_API_KEY")
NIM_MODEL  = os.environ.get("NIM_MODEL")

TOP_K          = int(os.environ.get("TOP_K", 5))
SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", 0.40))
MAX_TOKENS     = int(os.environ.get("MAX_TOKENS", 1024))
TEMPERATURE    = float(os.environ.get("TEMPERATURE", 0.20))

HF_EMBEDDING_URL = (
    "https://api-inference.huggingface.co/pipeline/feature-extraction/"
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Tambahkan baris ini
HF_RERANK_URL = "https://api-inference.huggingface.co/models/cross-encoder/ms-marco-MiniLM-L-12-v2"
HF_TOKEN = os.environ.get("HF_TOKEN", "") # Opsional, tapi disarankan

# ──────────────────────────────────────────────
# GUARDRAIL KEYWORDS
# ──────────────────────────────────────────────
SKRIPSI_KEYWORDS = [
    "skripsi", "proposal", "bimbingan", "pembimbing", "sidang", "seminar",
    "ujian", "penguji", "judul", "bab", "abstrak", "pendahuluan",
    "metodologi", "pembahasan", "kesimpulan", "daftar pustaka", "sitasi",
    "plagiasi", "turnitin", "format", "penulisan", "usu",
    "universitas sumatera utara", "ilmu komputer", "kurikulum", "sks",
    "penelitian", "data", "metode", "algoritma", "program", "sistem",
    "aplikasi", "analisis", "latar belakang", "rumusan masalah", "tujuan",
    "manfaat", "batasan", "tinjauan pustaka", "kerangka teori",
    "pengumpulan data", "implementasi", "pengujian", "validasi",
    "dosen", "mahasiswa", "akademik", "formulir", "pendaftaran",
    "nilai", "kkp", "kerja praktik", "flowchart", "diagram", "database",
    "jadwal", "timeline", "revisi",
]


def is_on_topic(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in SKRIPSI_KEYWORDS)

# ──────────────────────────────────────────────
# API — HUGGINGFACE RERANKING
# ──────────────────────────────────────────────
def rerank_hf(query: str, docs: list, top_k: int) -> list:
    if not docs:
        return []
    
    # Format input pasangan teks untuk pipeline text-classification di HuggingFace
    inputs = [{"text": query, "text_pair": d["text"]} for d in docs]
    
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
        
    try:
        resp = requests.post(
            HF_RERANK_URL,
            headers=headers,
            json={"inputs": inputs, "options": {"wait_for_model": True}},
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        
        # Ekstrak skor dari respons HuggingFace 
        # Format respons umumnya: [[{'label': 'LABEL_0', 'score': 0.85}], [...]]
        for i, result in enumerate(data):
            score = 0
            if isinstance(result, list) and len(result) > 0:
                score = result[0].get("score", 0)
            elif isinstance(result, dict):
                score = result.get("score", 0)
                
            docs[i]["rerank_score"] = float(score)
            
        # Urutkan berdasarkan skor tertinggi
        docs.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        
        # Kembalikan hanya sejumlah top_k terbaik
        return docs[:top_k]
        
    except Exception as e:
        app.logger.error(f"HF Reranking error: {e}")
        # Fallback: jika gagal, kembalikan dokumen asli dari Qdrant
        return docs[:top_k]
    
# ──────────────────────────────────────────────
# AUTH
# ──────────────────────────────────────────────
def hash_password(pwd: str) -> str:
    return hashlib.sha256(pwd.encode()).hexdigest()


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("authenticated"):
            if request.is_json:
                return jsonify({"error": "Unauthorized"}), 401
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated


@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        pwd      = data.get("password", "")
        username = data.get("username", "")

        if (not username) or (not pwd):
            return jsonify({
                "success": False,
                "error": "Username / Password wajib diisi."
            }), 400

        if hash_password(pwd) == hash_password(APP_PASSWORD) and hash_password(username) == hash_password(APP_USERNAME):
            session["authenticated"] = True
            session["username"] = username[:50]  # batasi panjang, tidak ada DB
            session["attempt_count"] = 0
            return jsonify({"success": True})
        else:
            attempts = session.get("attempt_count", 0) + 1
            session["attempt_count"] = attempts
            return jsonify({
                "success": False,
                "attempts": attempts,
                "locked": attempts >= 5
            }), 401
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))


# ──────────────────────────────────────────────
# PAGES
# ──────────────────────────────────────────────
@app.route("/")
@login_required
def index():
    return render_template("index.html", username=session.get("username", "Pengguna"))


# ──────────────────────────────────────────────
# API — EMBEDDING
# ──────────────────────────────────────────────
def get_embedding(text: str):
    try:
        resp = requests.post(
            HF_EMBEDDING_URL,
            headers={"Content-Type": "application/json"},
            json={"inputs": text, "options": {"wait_for_model": True}},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        # HuggingFace returns list-of-list; flatten to 1-D
        vec = data[0] if isinstance(data[0], list) else data
        return vec
    except Exception as e:
        app.logger.error(f"Embedding error: {e}")
        return None


# ──────────────────────────────────────────────
# API — QDRANT SEARCH
# ──────────────────────────────────────────────
def search_qdrant(vector, top_k=None, threshold=None, collection=None):
    url   = QDRANT_URL
    key   = QDRANT_API_KEY
    col   = collection or QDRANT_COLLECTION
    top_k = top_k or TOP_K
    thr   = threshold if threshold is not None else SCORE_THRESHOLD

    if not url or not vector:
        return []

    try:
        resp = requests.post(
            f"{url}/collections/{col}/points/search",
            headers={"Content-Type": "application/json", "api-key": key},
            json={
                "vector": vector,
                "limit": top_k,
                "score_threshold": thr,
                "with_payload": True,
            },
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json().get("result", [])
        return [
            {
                "text":   r["payload"].get("text") or r["payload"].get("content", ""),
                "source": r["payload"].get("source") or r["payload"].get("filename", "Dokumen"),
                "page":   r["payload"].get("page") or r["payload"].get("page_number", ""),
                "score":  round(r["score"], 4),
            }
            for r in results
        ]
    except Exception as e:
        app.logger.error(f"Qdrant error: {e}")
        return []


# ──────────────────────────────────────────────
# API — NIM LLM
# ──────────────────────────────────────────────
def call_llm(messages, max_tokens=None, temperature=None):
    key   = NIM_API_KEY
    model = NIM_MODEL
    base  = NIM_URL

    if not key:
        return {"text": "⚠️ NIM API Key belum dikonfigurasi di server (.env).", "error": True}

    try:
        resp = requests.post(
            f"{base}/chat/completions",
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {key}",
            },
            json={
                "model":       model,
                "messages":    messages,
                "max_tokens":  max_tokens or MAX_TOKENS,
                "temperature": temperature if temperature is not None else TEMPERATURE,
                "stream":      False,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return {"text": data["choices"][0]["message"]["content"], "error": False}
    except requests.HTTPError as e:
        err_body = {}
        try:
            err_body = e.response.json()
        except Exception:
            pass
        msg = err_body.get("error", {}).get("message", str(e))
        app.logger.error(f"LLM HTTP error: {msg}")
        return {"text": f"⚠️ Error dari NIM API: {msg}", "error": True}
    except Exception as e:
        app.logger.error(f"LLM error: {e}")
        return {"text": f"⚠️ Gagal menghubungi NIM API: {e}", "error": True}


# ──────────────────────────────────────────────
# BUILD SYSTEM PROMPT
# ──────────────────────────────────────────────
def build_system_prompt(docs: list, extra_prompt: str = "", bahasa: bool = True) -> str:
    lang_note = "Selalu jawab dalam Bahasa Indonesia yang baik dan benar." if bahasa else ""
    extra = f"\n\nInstruksi tambahan: {extra_prompt}" if extra_prompt else ""

    doc_block = ""
    if docs:
        parts = []
        for i, d in enumerate(docs):
            label = f"[Dok {i+1}"
            if d.get("source"):
                label += f" — {d['source']}"
            if d.get("page"):
                label += f", hal. {d['page']}"
            label += "]"
            parts.append(f"{label}\n{d['text']}")
        doc_block = "\n\nKONTEKS DOKUMEN PANDUAN SKRIPSI:\n" + "\n\n---\n\n".join(parts)

    return f"""Anda adalah asisten akademik khusus untuk mahasiswa Program Studi Ilmu Komputer \
Universitas Sumatera Utara (USU). Tugas Anda adalah membantu mahasiswa memahami dan menjalankan \
proses pengerjaan skripsi sesuai panduan resmi USU.

ATURAN PENTING:
1. Jawab HANYA pertanyaan yang berkaitan dengan skripsi, penelitian, akademik, atau panduan USU.
2. Jika pertanyaan tidak relevan, tolak dengan sopan dan arahkan ke topik skripsi.
3. Gunakan informasi dari dokumen yang diberikan sebagai referensi utama.
4. Jika informasi tidak tersedia di dokumen, akui keterbatasan dan sarankan menghubungi dosen/jurusan.
5. Berikan jawaban yang akurat, lengkap, dan terstruktur. {lang_note}{doc_block}{extra}"""


# ──────────────────────────────────────────────
# MAIN CHAT ENDPOINT
# ──────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
@login_required
def chat():
    body = request.get_json(silent=True) or {}
    query       = (body.get("query") or "").strip()
    history     = body.get("history", [])          # list of {role, content}
    use_memory  = body.get("memory", True)
    use_guardrail = body.get("guardrail", True)
    show_sources = body.get("show_sources", True)
    bahasa      = body.get("bahasa", True)
    extra_prompt = body.get("extra_prompt", "")
    top_k       = body.get("top_k", TOP_K)
    threshold   = body.get("threshold", SCORE_THRESHOLD)
    max_tokens  = body.get("max_tokens", MAX_TOKENS)
    temperature = body.get("temperature", TEMPERATURE)

    if not query:
        return jsonify({"error": "Query tidak boleh kosong."}), 400

    # ── Guardrail ──
    if use_guardrail and not is_on_topic(query):
        return jsonify({
            "answer":    (
                "Mohon maaf, saya hanya bisa membantu pertanyaan seputar **panduan skripsi** "
                "dan **akademik** di Program Studi Ilmu Komputer USU.\n\n"
                "Silakan tanyakan hal-hal seperti:\n"
                "- Proses pengajuan judul skripsi\n"
                "- Format dan struktur penulisan\n"
                "- Prosedur bimbingan dan seminar\n"
                "- Persyaratan sidang skripsi\n"
                "- Panduan penulisan daftar pustaka"
            ),
            "sources":   [],
            "guardrail": True,
            "error":     False,
        })

    # ── Embedding ──
    vector = get_embedding(query)

    # ── Qdrant Search & HF Reranking ──
    docs = []
    if vector:
        # Ambil dokumen 3x lebih banyak dari Qdrant untuk di-rerank
        fetch_k = max(15, top_k * 3)
        initial_docs = search_qdrant(vector, top_k=fetch_k, threshold=threshold)

        if initial_docs:
            # Lakukan reranking via HF API dan langsung potong ke top_k
            docs = rerank_hf(query, initial_docs, top_k)

    # ── Build LLM Messages ──
    system_prompt = build_system_prompt(docs, extra_prompt=extra_prompt, bahasa=bahasa)
    messages = [{"role": "system", "content": system_prompt}]

    if use_memory and history:
        # kirim maksimal 10 pesan terakhir sebagai konteks
        for m in history[-10:]:
            if m.get("role") in ("user", "assistant"):
                messages.append({"role": m["role"], "content": m["content"]})

    messages.append({"role": "user", "content": query})

    # ── Call LLM ──
    result = call_llm(messages, max_tokens=max_tokens, temperature=temperature)

    return jsonify({
        "answer":    result["text"],
        "sources":   docs if show_sources else [],
        "guardrail": False,
        "error":     result["error"],
    })


# ──────────────────────────────────────────────
# HEALTH CHECK
# ──────────────────────────────────────────────
@app.route("/api/health")
@login_required
def health():
    return jsonify({
        "status":     "ok",
        "qdrant_url": bool(QDRANT_URL),
        "nim_key":    bool(NIM_API_KEY),
        "model":      NIM_MODEL,
        "collection": QDRANT_COLLECTION,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
