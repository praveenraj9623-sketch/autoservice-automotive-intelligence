import io

import pdfplumber
import streamlit as st
from rank_bm25 import BM25Okapi

st.set_page_config(
    page_title="AutoService Intelligence",
    page_icon="🚗",
    layout="wide",
)

st.title("AutoService Intelligence")
st.caption("Automotive manual search | Prototype — in progress")

st.info(
    "Implemented: PDF text extraction, keyword retrieval, and PDF page references. "
    "Planned: semantic search, hybrid fusion, reranking, and grounded Q&A."
)

st.write(
    "Upload a vehicle manual and search its pages. Results apply only to "
    "the uploaded document. This is an independent portfolio prototype."
)

@st.cache_data(max_entries=2, show_spinner=False)
def extract_pages(content):
    records = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        if len(pdf.pages) > 600:
            raise ValueError("Maximum supported length is 600 pages.")
        for number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                records.append((number, text))
    return records

uploaded = st.file_uploader("Upload a text-based vehicle manual", type=["pdf"])

if uploaded is not None:
    if uploaded.size > 20 * 1024 * 1024:
        st.error("Use a PDF smaller than 20 MB.")
        st.stop()

    try:
        with st.spinner("Extracting pages..."):
            pages = extract_pages(uploaded.getvalue())
    except Exception:
        st.error("Could not process this PDF. Try a smaller text-based document.")
        st.stop()

    if not pages:
        st.warning("No readable text found. Scanned-document OCR is not implemented.")
        st.stop()

    st.success(f"Text extracted from {len(pages)} pages.")

    question = st.text_input(
        "Search the manual",
        placeholder="For example: tyre pressure warning",
        max_chars=300,
    )

    if question.strip():
        tokenized = [text.lower().split() for _, text in pages]
        query_tokens = question.lower().split()
        vocabulary = set(word for page in tokenized for word in page)

        if not any(word in vocabulary for word in query_tokens):
            st.warning("No matching keywords found. Try another phrase.")
        else:
            scores = BM25Okapi(tokenized).get_scores(query_tokens)
            candidates = [
                i for i, tokens in enumerate(tokenized)
                if set(query_tokens).intersection(tokens)
            ]
            ranked = sorted(
                candidates, key=lambda i: float(scores[i]), reverse=True
            )[:5]

            for i in ranked:
                number, text = pages[i]
                with st.expander(f"PDF page {number}", expanded=True):
                    st.text(text)

st.divider()
st.markdown(
    "Foundation: [MongoDB Car Manual Explorer]"
    "(https://github.com/mongodb-industry-solutions/"
    "manufacturing-car-manual-RAG). "
    "Upstream license and notices are retained."
)
st.caption(
    "This Streamlit interface currently uses keyword search. "
    "It does not yet run the upstream Atlas pipeline or generate answers."
)
