# Importing dependencies 
from helper_utils import word_wrap
from pypdf import PdfReader # To read PDF files
import os # Interact with the operating system
from groq import Groq # Imports Groq's API to interact with models
import chromadb # Database for storing and querying embeddings generated from the text data
from dotenv import load_dotenv # To load the variables from .env to manage API keys securely
import numpy as np
import umap

load_dotenv()
groq_key = os.getenv("API_KEY")
client = Groq(api_key = groq_key)

# Reading the pdf file using PdfReader - Number of pages = 116
reader = PdfReader("E:/Personal Project/rags_query_expansion/data/microsoft-annual-report.pdf")

# Looping through each page, extracting the text and stripping extra spaces.
pdf_texts = []
for p in reader.pages:
    text = p.extract_text().strip()
    pdf_texts.append(text)

# print(len(pdf_texts))   Output - 116

# Filtering out empty strings (avoiding meaningless data)
# filtered_pdf = []
# for text in pdf_texts:
#     if text:
#         filtered_pdf.append(text)
pdf_texts = [text for text in pdf_texts if text] # 115 (One empty page removed)

# Testing - Printing the contents of one page
# print(
#     word_wrap(pdf_texts[0], width = 100)
# )

# Splitting the texts into chucks 
from langchain.schema import Document
from langchain.text_splitter import(         # Importing langchain 
    RecursiveCharacterTextSplitter,          # Splits text by multiple levels of seperators
    SentenceTransformersTokenTextSplitter    # Splits text based on tokens
)

# Recursive Character Splitter
character_splitter = RecursiveCharacterTextSplitter(
    separators = ["\n\n", "\n", ". ", " ", ""],     # Breaks text by paragraphs, newlines, sentences and spaces
    chunk_size = 1000,      # Each chuck will be a 1000 characters
    chunk_overlap = 0       # No overlaps between chucks
)

# Combining all the elements of the pdf_texts list into one string but seperated by \n\n.
full_document = "\n\n".join(pdf_texts)

# Wrapping the string into a Document object before splitting it
document = Document(page_content = full_document)

# Splitting into chucks based on character length 
character_split_texts = character_splitter.split_documents([document]) # Splitting the string into chucks of size 1000

# print(character_split_texts[0])
print(f"Total Number of chucks based on character splitting: {len(character_split_texts)}") # Number of chucks - 350

# Splitting these character-based chunks further into token-based chunks
token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap = 0,
    tokens_per_chunk = 256
)

# Splits the chuck according to the tokens and sends the rest of the tokens to the next chunk
token_split_texts = []
# Passing one string(page_content from the Document) at a time
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text.page_content) 

# print(token_split_texts[0])
print(f"Number of chucks after token based splitting: {len(token_split_texts)}") # 359 from 350

#Using embedding function from chromadb to convert words into vectors
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

embedding_function = SentenceTransformerEmbeddingFunction()
# Uses pre-trained models for embeddings from the sentence-transformers library

#print(embedding_function([token_split_texts[0]]))

chroma_client = chromadb.Client() # Interact with the Chroma database
chroma_collection = chroma_client.create_collection(
    "microsoft_collection", embedding_function = embedding_function
) 

# Creating embeddings for documents
ids = []                                # unique identifier for each chunk
for i in range(len(token_split_texts)):
    ids.append(str(i))

chroma_collection.add(ids = ids, documents = token_split_texts) # Adding documents and embeddings into the database
# The embeddings for these texts are automatically generated using the embedding_function.

# checking the number of documents in the collection
count = chroma_collection.count()
# print(f"Number of collections: {count}")

# query = "What was the total revenue for the year?"
# result = chroma_collection.query(query_texts = [query], n_results = 5)
# retrieved_documents = result["documents"][0]

# for docs in retrieved_documents:
#     print(word_wrap(docs))
#     print("\n")

# Generating Augumented queries - Query expansion
def gen_augument_query(query, model = "llama3-8b-8192"):
    # Propmt Engineering to guide the model in generating meaningful questions.
    # Elements - Role, context, instruction, format
    prompt = """
            You are a knowledgeable financial research assistant. 
            Your users are inquiring about an annual report. 
            For the given question, propose up to five related questions to assist them in finding the information they need. 
            Provide concise, single-topic questions (without compounding sentences) that cover various aspects of the topic. 
            Ensure each question is complete and directly related to the original inquiry. 
            List each question on a separate line without numbering.
            """
    
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query}
    ]

    # API call - sending the request
    response = client.chat.completions.create(
        model = model, 
        messages = messages
    )

    # Processing the response
    content = response.choices[0].message.content
    content = content.split("\n")
    return content

original_query = ("What details can you provide about the factors that led to revenue growth?")

# Generating augumented queries and printing them
aug_queries = gen_augument_query(original_query)
print("The augumented Queries are:")
for query in aug_queries:
    print("\n- ", query)

# Combining original query with augumented queries
all_queries = [original_query] + aug_queries

# chroma_collection.query performs a similarity search
answers = chroma_collection.query(
    query_texts = all_queries,
    n_results = 5,                          # Returns the top 5 results
    include = ["documents", "embeddings"]   # returns both (actual text of the chunk, their vector representations)
)

# Retrieving just the documents(answer to each query)
retrieved_results = answers["documents"][0]

# Deduplicating the retrieved answer
unique_docs = set()
for documents in retrieved_results:
    for doc in documents: 
        unique_docs.add(doc)

# Printing the retrieved answer for each query
print("")
for i, docs in enumerate(retrieved_results):
    print(f"Query: {all_queries[i]}")
    print("")
    print("Result:")
    print(docs)
    print("-" * 100)

# Visualising the embeddings in a lower-dimensional space - to identify meaningful patterns
import umap

# Projecting the embedding into a lower dimensional space using UMAP transformer.
def project_embeddings(embeddings, umap_transform):
    return umap_transform.transform(embeddings)

# Embeddings for the entire dataset in the vector space
data = chroma_collection.get(include=["embeddings"])    #.get() retrieves the dictionary from collection
embeddings = data["embeddings"]     # Accessing only the embeddings from the dictionary

# Dimensionality reduction using UMAP (Uniform Manifold Approximation and Projection)
umap_transform = umap.UMAP(random_state = 0, transform_seed = 0).fit(embeddings)

# Reducing the dimensions of the document embeddings
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform )

# Converting the queries into embeddings to compare them directly with the document embeddings
original_query_embeddings = embedding_function([original_query])
augumented_query_embeddings = embedding_function(all_queries)

# Projecting the query embeddings into the same UMAP space as the dataset
project_original_query = project_embeddings(original_query_embeddings, umap_transform)
project_augumented_query = project_embeddings(augumented_query_embeddings, umap_transform)

retrieved_embeddings = answers["embeddings"]
result_embeddings = [item for sublist in retrieved_embeddings for item in sublist]
project_answers_embedding = project_embeddings(result_embeddings, umap_transform)

# Ideally, the queries should be close to relevant document clusters.

# Visualising the embeddings
import matplotlib.pyplot as plt

# Plotting all the data embeddings in the dataset into a 2D space
plt.scatter(
    projected_dataset_embeddings[:, 0],     
    projected_dataset_embeddings[:, 1],     
    s = 10,                                 # Size of dots
    facecolors = "none",
    color = "blue"
)

# Plotting the original query - Shows the search results
plt.scatter(
    project_original_query[:, 0],
    project_original_query[:, 1],
    s = 150,
    marker = "X",
    color = "r",
)

# Plotting the Augumented queries - shows the rephrased query
plt.scatter(
    project_augumented_query[:, 0],
    project_augumented_query[:, 1],
    s = 150, 
    marker = "X",
    color = "orange"
)

# Plotting the retrieved documents - Shows the search results
plt.scatter(
    project_answers_embedding[:, 0],
    project_answers_embedding[:, 1],
    s = 100, 
    facecolors = "none",    # Hollow circle
    edgecolors = "g",       
)

plt.gca().set_aspect("equal", "datalim") 
plt.title(f"{original_query}")
plt.axis("off")
plt.show()