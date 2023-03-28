from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai.embeddings_utils import get_embedding, cosine_similarity
from parameters import *
import configparser
import pandas as pd
import numpy as np
import gradio as gr
import tiktoken
import logging
import pickle
import openai
import time
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def num_tokens_from_string(string: str, encoding_name="cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
 
class GPT():
    
    def __init__(self) -> None:
        config_file = 'config.ini'
        with open(config_file) as f:
            config = configparser.ConfigParser()
            config.read_file(f)
        os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY') or config.get('PERSONAL', 'OPENAI_API_KEY')
        openai.api_key = os.environ["OPENAI_API_KEY"]
    
    def generate_docs(self, file_name):
        data = []
        if not os.path.exists(f"chunks/{file_name}.pkl"):
            loader = UnstructuredPDFLoader(f'docs/{file_name}.pdf')
            data = loader.load()
            logger.info(f'You have {len(data)} document(s) in your data')
            logger.info(f'There are {len(data[0].page_content)} characters in your document')
        
        return data

    def chunk_data(self, data, chunk_file, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        file = f'chunks/{chunk_file}.pkl'
        if os.path.exists(file):
            with open(file, 'rb') as f:
                chunks = pickle.load(f)
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = text_splitter.split_documents(data)
            logger.info(f'Now you have {len(chunks)} documents')
            with open(f'chunks/{chunk_file}.pkl', 'wb') as f:
                pickle.dump(chunks, f)

        return chunks
    
    def obtain_embedding(self, chunk_number, page_content, engine):
        logger.info(f"Chunk number: {chunk_number}")
        return get_embedding(page_content, engine)

 
    def get_df_embedded(self, chunks, vector_name):
        logging.info(f'Embedding total of chunks: {len(chunks)}')
        file = f'vectors/{vector_name}.pkl'
        if not os.path.exists(file):
            df = pd.DataFrame([{'page_content':chunk.page_content, 
                                'tokens': num_tokens_from_string(chunk.page_content),
                                'embedding': self.obtain_embedding(chunk_number, chunk.page_content, engine=EMBEDDING_MODEL)} 
                                                                                for chunk_number, chunk in enumerate(chunks)])
            df.to_pickle(f'vectors/{vector_name}.pkl')
        
    def vector_similarity(self, query_embedding, doc_embedding):
        return cosine_similarity(doc_embedding, query_embedding)

    def obtain_similar_vectors(self, query, embeddings):
        query_embedding = get_embedding(query, engine=EMBEDDING_MODEL)        
        document_similarities = sorted([
            (self.vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in embeddings['embedding'].to_dict().items()
        ], reverse=True)
    
        return document_similarities
    
    def construct_prompt(self, question: str, document_similarities: dict, vectors_df: pd.DataFrame) -> str:
        encoding = tiktoken.encoding_for_model(COMPLETIONS_MODEL)
        separator_len = len(encoding.encode(SEPARATOR))        
        chosen_sections = []
        chosen_sections_len = 0
        chosen_sections_indexes = []
        extra_tokens = num_tokens_from_string(question) + num_tokens_from_string(HEADER) + separator_len
        max_section_len = MAX_SECTION_LEN - extra_tokens
        logging.info(f"Maximum prompt's tokens: {max_section_len}")
        
        for _, section_index in document_similarities:
            document_section = vectors_df.loc[section_index]            
            chosen_sections_len += document_section.tokens + extra_tokens
            if chosen_sections_len > max_section_len:
                break
                
            chosen_sections.append(SEPARATOR + document_section.page_content.replace("\n", " "))
            chosen_sections_indexes.append(str(section_index))
                
        logger.info(f"Selected {len(chosen_sections)} document sections:")
        logger.info("\n".join(chosen_sections_indexes))
        
        header = f"""{HEADER}\n\nContext:\n""" 
        prompt = header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"
    
        logger.info(f'Number of tokens to use: {num_tokens_from_string(prompt)}')
        return prompt
    
    def answer_query_with_context(self, prompt, show_prompt: bool = False):
        if show_prompt:
            logger.info(prompt)
            
        response = openai.ChatCompletion.create(
                    messages=[{"role": "user", "content": prompt}],
                    **COMPLETIONS_API_PARAMS
                )

        return response["choices"][0]["message"]["content"].strip(" \n")           

    def load_vectorsDB(self, files, path):
        dfs = []
        for f in files:
            if f.endswith('.pkl'):
                logging.info(f'Loading vector: {f}')
                df = pd.read_pickle(os.path.join(path, f))
                dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def load_vectorDB(self, vectors=[], path='./vectors'):
        files = os.listdir(path) if vectors == [] else vectors
        self.vectorDB = self.load_vectorsDB(files, path)

    def preprocess(self, file_name):
        documents = self.generate_docs(file_name)
        chunks = self.chunk_data(documents, chunk_file=file_name)
        self.get_df_embedded(chunks, vector_name=file_name)

    def query(self, query):
        document_similarities = self.obtain_similar_vectors(query, self.vectorDB)
        prompt = self.construct_prompt(query, document_similarities, self.vectorDB)
        answer = self.answer_query_with_context(prompt)
        return answer

if __name__ == '__main__':
    file_names = ['DeliveringHappiness', 
                  'Handbook-ExperienceEconomyPastPresentandFuture',
                  'Pine_Gilmore_The_experience_economy_1999']
    preprocessData = GPT()
    [preprocessData.preprocess(file_name) for file_name in file_names]
