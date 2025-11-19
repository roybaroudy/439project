import weaviate
from weaviate.classes.init import Auth
import weaviate.classes as wvc
import dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pdfminer.high_level import extract_text
import json

dotenv.load_dotenv()
WEAVIATE_URL = dotenv.get_key(dotenv.find_dotenv(), "WEAVIATE_URL")
WEAVIATE_API_KEY = dotenv.get_key(dotenv.find_dotenv(), "WEAVIATE_API_KEY")

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,          
    auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
)

try:
    client.collections.delete("PDFChunk")
except Exception:
    pass

pdf_chunks = client.collections.create(
    name="PDFChunk",
    properties=[
        wvc.config.Property(name="chunk_id", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT),
    ],
    vector_config=wvc.config.Configure.Vectors.self_provided(),  # <- we pass our own vectors
)

def extract_pdf_text(pdf_path):
    full_text = extract_text(pdf_path)
    with open(r"C:\Users\royba\Downloads\EECE-439\project\data\text.txt", "w", encoding="utf-8") as f:
        f.write(full_text)
    return full_text

def chunk_pdf_text(pdf_text, chunk_size=1000, overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_text(pdf_text)
    return [{"chunk_id": f"chunk_{i+1}", "content": c} for i, c in enumerate(chunks)]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def store_chunks_in_weaviate(chunks):
    collection = client.collections.get("PDFChunk")

    with collection.batch.dynamic() as batch:
        for chunk in chunks:
            vector = embeddings.embed_query(chunk["content"])
            batch.add_object(
                properties={
                    "chunk_id": chunk["chunk_id"],
                    "content": chunk["content"],
                },
                vector=vector,
            )

def save_chunks_to_file(chunks, filename="chunks.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


pdf_path = r"C:\Users\royba\Downloads\EECE-439\project\data\ISO_IEC_42001_2023en.pdf"
pdf_text = extract_pdf_text(pdf_path)
chunks = chunk_pdf_text(pdf_text)
store_chunks_in_weaviate(chunks)



