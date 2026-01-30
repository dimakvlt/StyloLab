# utils/processing.py
import re
from typing import List
try:
    import nltk
    from nltk import word_tokenize, sent_tokenize
    nltk.download("punkt", quiet=True)
    HAVE_NLTK = True
except Exception:
    HAVE_NLTK = False

def load_txt_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    raw = uploaded_file.read()
    if raw is None:
        return ""
    if isinstance(raw, (bytes, bytearray)):
        try:
            return raw.decode("utf-8")
        except Exception:
            try:
                return raw.decode("latin-1")
            except Exception:
                return raw.decode("utf-8", errors="ignore")
    return str(raw)

_nonalpha_re = re.compile(r"[^a-zA-Zà-žÀ-Ž0-9'\s]")

def clean_text(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\r\n", " ").replace("\n", " ")
    t = t.lower()
    t = re.sub(_nonalpha_re, " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def clean_text_basic(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\r\n", " ").replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def clean_and_tokenize(text: str) -> List[str]:
    if not text:
        return []
    t = clean_text(text)
    try:
        if HAVE_NLTK:
            toks = word_tokenize(t)
        else:
            toks = re.findall(r"\w+['-]?\w*|\w+", t)
    except Exception:
        toks = re.findall(r"\w+['-]?\w*|\w+", t)
    toks = [tok for tok in toks if re.search(r"\w", tok)]
    return toks

def tokenize_with_choice(text: str, choice: str):
    if text is None:
        return []
    choice = (choice or "").lower()
    if "whitespace" in choice:
        return [t for t in text.split() if t.strip()]
    if "regex" in choice:
        return re.findall(r"\w+['-]?\w*|\w+", text.lower())
    if "spacy" in choice:
        try:
            import spacy
            for model in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"):
                try:
                    nlp = spacy.load(model)
                    doc = nlp(text)
                    return [tok.text for tok in doc]
                except Exception:
                    continue
            nlp = spacy.blank("en")
            doc = nlp(text)
            return [tok.text for tok in doc]
        except Exception:
            pass
    if "unicode" in choice:
            return re.findall(r"\p{L}+", text.lower())
    if "char_ngrams" in choice:
        text = re.sub(r"\s+", " ", text.lower())
        n = 3
        return [text[i:i+n] for i in range(len(text)-n+1)]

    
    return clean_and_tokenize(text)

def chunk_words(tokens: List[str], size: int = 2000) -> List[List[str]]:
    if not tokens:
        return []
    return [tokens[i:i+size] for i in range(0, len(tokens), size)]

def chunk_text(text: str, chunk_size: int = 2000) -> List[str]:
    if not text:
        return []
    try:
        toks = word_tokenize(text) if HAVE_NLTK else re.findall(r"\w+['-]?\w*|\w+", text.lower())
    except Exception:
        toks = re.findall(r"\w+['-]?\w*|\w+", text.lower())
    chunks = []
    for i in range(0, len(toks), chunk_size):
        chunk = " ".join(toks[i:i+chunk_size])
        chunks.append(chunk)
    return chunks
# -------------------------------
# DOCUMENT EXTRACTORS
# -------------------------------
import pdfplumber
from docx import Document

def extract_text_from_pdf(file_obj):
    text = []
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)

def extract_text_from_docx(file_obj):
    doc = Document(file_obj)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)

def extract_text_file(file_obj, filename):
    name = filename.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(file_obj)
    elif name.endswith(".docx"):
        return extract_text_from_docx(file_obj)
    elif name.endswith(".txt"):
        return file_obj.read().decode("utf-8")
    else:
        raise ValueError("Unsupported file format: only PDF, DOCX, TXT allowed")
