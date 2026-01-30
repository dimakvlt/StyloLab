import streamlit as st
from utils.processing import extract_text_file

def render_inputs(analysis_mode):
    if analysis_mode.startswith("Compare"):
        colA, colB, colU = st.columns(3)

        with colA:
            fileA = st.file_uploader("Author A", type=["txt","pdf","docx"])
            pastedA = st.text_area("Paste A", height=220)

        with colB:
            fileB = st.file_uploader("Author B", type=["txt","pdf","docx"])
            pastedB = st.text_area("Paste B", height=220)

        with colU:
            fileU = st.file_uploader("Unknown", type=["txt","pdf","docx"])
            pastedU = st.text_area("Paste U", height=220)
    else:
        fileU = st.file_uploader("Upload text", type=["txt","pdf","docx"])
        pastedU = st.text_area("Paste text", height=300)
        fileA = fileB = pastedA = pastedB = None

    return fileA, pastedA, fileB, pastedB, fileU, pastedU


def get_text(fileobj, pasted):
    if fileobj:
        return extract_text_file(fileobj, fileobj.name)
    if pasted and pasted.strip():
        return pasted
    return ""
