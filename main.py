# main.py

from langchain_community.document_loaders.notiondb import NotionDBLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tiktoken
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import logging
import time
from dotenv import load_dotenv
import os
from llama_index.readers.notion import NotionPageReader
from langchain.schema import Document as LangChainDocument
import hashlib
from uuid import uuid4
from enum import Enum
from datetime import datetime, timedelta
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("uvicorn")

# Load environment variables
load_dotenv()

NOTION_TOKEN = os.getenv('NOTION_TOKEN')
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP'))
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX = os.getenv('PINECONE_INDEX')

# Global variables
docs = []
length_of_docs = 0
total_cost = 0
total_embedding_cost = 0

class CleanupMode(str, Enum):
    NONE = "None"
    INCREMENTAL = "Incremental"
    FULL = "Full"

class UpsertRequest(BaseModel):
    notion_id: str
    doc_type: str
    cleanup_mode: CleanupMode = CleanupMode.INCREMENTAL
    last_update_time: str = None

def generate_source_id(content, metadata):
    """Generate a unique source ID based on content and metadata."""
    unique_string = f"{content}{str(metadata)}".encode('utf-8')
    return hashlib.md5(unique_string).hexdigest()

def convert_llamaindex_to_langchain(llamaindex_doc):
    """Convert a LlamaIndex document to a LangChain document."""
    content = llamaindex_doc.text
    langchain_doc = LangChainDocument(
        page_content=content,
        metadata=llamaindex_doc.metadata
        if hasattr(llamaindex_doc, 'metadata') else {})
    return langchain_doc

async def load_documents_from_notion_db(document_id, last_update_time=None):
    logger.info("Loading documents from notion database")
    start_time = time.time()

    loader = NotionDBLoader(
        integration_token=NOTION_TOKEN,
        database_id=document_id,
        request_timeout_sec=60,
    )

    global docs
    raw_docs = loader.load()
    docs = []
    for doc in raw_docs:
        if last_update_time is None or doc.metadata.get('last_edited_time', '') > last_update_time:
            source_id = generate_source_id(doc.page_content, doc.metadata)
            doc.metadata['source_id'] = source_id
            docs.append(doc)

    global length_of_docs
    length_of_docs = len(docs)

    process_time = time.time() - start_time
    logger.info(
        f"{length_of_docs} documents loaded - Duration: {process_time:.4f} seconds"
    )

async def load_documents_from_notion_page(document_id, last_update_time=None):
    logger.info("Loading documents from notion page")
    start_time = time.time()

    reader = NotionPageReader(integration_token=NOTION_TOKEN)
    documents = reader.load_data(page_ids=[document_id])

    global docs
    docs = []
    for document in documents:
        langchain_doc = convert_llamaindex_to_langchain(document)
        if last_update_time is None or langchain_doc.metadata.get('last_edited_time', '') > last_update_time:
            source_id = generate_source_id(langchain_doc.page_content, langchain_doc.metadata)
            langchain_doc.metadata['source_id'] = source_id
            docs.append(langchain_doc)

    global length_of_docs
    length_of_docs = len(docs)

    process_time = time.time() - start_time
    logger.info(
        f"{length_of_docs} documents loaded - Duration: {process_time:.4f} seconds"
    )

async def split_documents():
    logger.info("Splitting documents into chunks")
    start_time = time.time()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    global docs
    chunked_docs = []
    for doc in docs:
        chunks = text_splitter.split_documents([doc])
        for chunk in chunks:
            chunk.metadata['source_id'] = doc.metadata['source_id']
            chunk.metadata['chunk_id'] = str(uuid4())  # Add a unique chunk ID
            chunked_docs.append(chunk)

    docs = chunked_docs
    global length_of_docs
    length_of_docs = len(docs)

    process_time = time.time() - start_time
    logger.info(
        f"{length_of_docs} documents in total after splitting - Duration: {process_time:.4f} seconds"
    )

async def calculate_pinecone_cost():
    logger.info("Calculating pinecone & embeddings cost")

    encoder = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    documents = [doc.page_content for doc in docs]

    def count_tokens(text, encoder):
        tokens = encoder.encode(text)
        return len(tokens)

    def calculate_embedding_cost(total_tokens, cost_per_token):
        return total_tokens * cost_per_token

    total_tokens = sum(count_tokens(doc, encoder) for doc in documents)
    total_vectors = len(documents)

    storage_cost_per_vector = 0.000001
    read_cost_per_vector = 0.000008
    write_cost_per_vector = 0.000002
    cost_per_token = 0.00000013

    storage_cost = total_vectors * storage_cost_per_vector
    read_cost = total_vectors * read_cost_per_vector
    write_cost = total_vectors * write_cost_per_vector
    global total_cost
    total_cost = storage_cost + read_cost + write_cost
    global total_embedding_cost
    total_embedding_cost = calculate_embedding_cost(total_tokens, cost_per_token)

async def get_existing_documents(vectorstore):
    """Retrieve existing documents from Pinecone."""
    results = await vectorstore.asimilarity_search("", k=10000)  # Adjust k as needed
    return {
        doc.metadata['source_id']: {
            'chunks': [
                chunk.metadata['chunk_id'] for chunk in results
                if chunk.metadata['source_id'] == doc.metadata['source_id']
            ]
        }
        for doc in results if 'source_id' in doc.metadata
    }

async def cleanup_and_upsert_documents(docs, cleanup_mode: CleanupMode):
    logger.info(f"Upserting documents to Pinecone with {cleanup_mode} cleanup mode")
    start_time = time.time()

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)
    vectorstore = PineconeVectorStore(pinecone_api_key=PINECONE_API_KEY,
                                      embedding=embeddings,
                                      index_name=PINECONE_INDEX)

    existing_docs = await get_existing_documents(vectorstore)

    chunks_to_delete = set()
    chunks_to_upsert = []
    for doc in docs:
        source_id = doc.metadata['source_id']
        if source_id in existing_docs:
            if cleanup_mode != CleanupMode.NONE:
                chunks_to_delete.update(existing_docs[source_id]['chunks'])
            if cleanup_mode == CleanupMode.INCREMENTAL:
                chunks_to_upsert.append(doc)
        else:
            chunks_to_upsert.append(doc)

    if cleanup_mode == CleanupMode.FULL:
        for source_id, doc_info in existing_docs.items():
            if source_id not in [doc.metadata['source_id'] for doc in docs]:
                chunks_to_delete.update(doc_info['chunks'])

    deleted_chunks = 0
    if chunks_to_delete:
        await vectorstore.adelete(ids=list(chunks_to_delete))
        deleted_chunks = len(chunks_to_delete)
        logger.info(f"Deleted {deleted_chunks} outdated chunks")

    upserted_chunks = 0
    if chunks_to_upsert:
        response = await vectorstore.aadd_documents(documents=chunks_to_upsert)
        upserted_chunks = len(chunks_to_upsert)
        logger.info(f"Upserted {upserted_chunks} document chunks")

    process_time = time.time() - start_time
    logger.info(f"Operation completed - Duration: {process_time:.4f} seconds")

    return {
        "deleted_chunks": deleted_chunks,
        "upserted_chunks": upserted_chunks,
        "total_chunks": len(docs),
        "deduplication_rate": 1 - (upserted_chunks / len(docs)) if len(docs) > 0 else 0,
        "process_time": process_time
    }

app = FastAPI(docs="/docs")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/health")
async def health_check():
    logger.info("checking health")
    return {"status": "healthy"}

@app.post("/upsert")
async def upsert(request: UpsertRequest):
    logger.info(f"Upsert function started with {request.cleanup_mode} cleanup mode")
    start_time = time.time()
    try:
        if request.last_update_time is None:
            request.last_update_time = (datetime.now() - timedelta(days=7)).isoformat()

        if request.doc_type == "database":
            await load_documents_from_notion_db(request.notion_id, request.last_update_time)
        elif request.doc_type == "page":
            await load_documents_from_notion_page(request.notion_id, request.last_update_time)
        else:
            raise HTTPException(status_code=400, detail="Invalid document type")

        await split_documents()
        await calculate_pinecone_cost()

        upsert_result = await cleanup_and_upsert_documents(docs, request.cleanup_mode)

        total_time = time.time() - start_time
        return {
            "success": True,
            "total_vectors": length_of_docs,
            "total_pinecone_cost": total_cost,
            "total_embedding_cost": total_embedding_cost,
            "upsert_details": upsert_result,
            "cleanup_mode": request.cleanup_mode,
            "last_update_time": request.last_update_time,
            "total_process_time": total_time
        }

    except Exception as error:
        logger.error(f"Error in upsert: {str(error)}")
        raise HTTPException(status_code=500, detail=str(error))

