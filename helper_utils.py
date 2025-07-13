import numpy as np
import chromadb
import pandas as pd
from pypdf import PdfReader

# def project_embeddings(embedding, umap_transform):
#     return

def word_wrap(text, width = 100): 
    """
    wraps the given text to the specific width and returns a string
    """
    return "\n".join([text[i: i + width] for i in range(0, len(text), width)])
    