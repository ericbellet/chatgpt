from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai.embeddings_utils import get_embedding, cosine_similarity
from typing import List, Dict, Tuple
import openai
import configparser
import pandas as pd
import numpy as np
import gradio as gr
import logging
import tiktoken
import pickle
import sys
import os

logger = logging.getLogger()
COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"

def num_tokens_from_string(string: str, encoding_name="cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
 
def search_reviews(df, product_description, n=3, pprint=True):
   embedding = get_embedding(product_description, model='text-embedding-ada-002')
   df['similarities'] = df.ada_embedding.apply(lambda x: cosine_similarity(x, embedding))
   res = df.sort_values('similarities', ascending=False).head(n)
   return res
 

#num_tokens_from_string("tiktoken is great!", "cl100k_base")

class GPT():
    
    def __init__(self) -> None:
        config_file = 'config.ini'
        with open(config_file) as f:
            config = configparser.ConfigParser()
            config.read_file(f)
        os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY') or config.get('PERSONAL', 'OPENAI_API_KEY')

        # self.iface = gr.Interface(fn=self.chatbot,
        #                           inputs=gr.inputs.Textbox(lines=7, label="Enter your text"),
        #                           output="text",
        #                           title="Test")
    
    def generate_docs_langchain(self, file_name):
        data = []
        if not os.path.exists(f"chunks/{file_name}.pickle"):
            loader = UnstructuredPDFLoader(f'docs/{file_name}.pdf')
            data = loader.load()
            logger.info(f'You have {len(data)} document(s) in your data')
            logger.info(f'There are {len(data[0].page_content)} characters in your document')
        
        return data

    def chunk_data_langchain(self, data, chunk_file, chunk_size=1000, chunk_overlap=0):
        file = f'chunks/{chunk_file}.pickle'
        if os.path.exists(file):
            with open(file, 'rb') as f:
                chunks = pickle.load(f)
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = text_splitter.split_documents(data)
            logger.info(f'Now you have {len(chunks)} documents')
            with open(f'chunks/{chunk_file}.pickle', 'wb') as f:
                pickle.dump(chunks, f)

        return chunks
    
    def get_df_embedded(self, chunks):
        df = pd.DataFrame([{'page_content':chunk.page_content, 
                            'tokens': num_tokens_from_string(chunk.page_content),
                            'embedding': self.get_embedding(chunk.page_content)} for chunk in chunks])
        return df

    
    
    def get_embedding(self, text: str, model: str=EMBEDDING_MODEL) -> List[float]:
        words = text.split()
        word_to_number = {word: i for i, word in enumerate(words)}
        vector = [word_to_number[word] for word in words]
        return vector
        # result = openai.Embedding.create(
        # model=model,
        # input=text
        # )
        # return result["data"][0]["embedding"]
    
    def vector_similarity(self, x: List[float], y: List[float]) -> float:
        """
        Returns the similarity between two vectors.
        
        Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
        """
        try:
            return np.dot(np.array(x), np.array(y))
        except:
            return np.dot(np.array(x), np.array(x)) #Remove this line

    def obtain_similar_vectors(self, query, embeddings):
        """
        Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
        to find the most relevant sections. 
        
        Return the list of document sections, sorted by relevance in descending order.
        """
        query_embedding = self.get_embedding(query) #Change to get_embedding
        query_embedding = [102, 33, 2, 3, 4, 5, 6, 7, 8, 9, 57, 11, 112, 13, 14, 15, 16, 80, 18, 19, 20, 21, 22, 23, 24, 25, 130, 28, 28, 130, 127, 128, 102, 33, 34, 35, 36, 73, 42, 39, 40, 41, 42, 43, 44, 45, 87, 47, 48, 49, 50, 83, 52, 87, 88, 89, 56, 57, 58, 59, 60, 70, 62, 87, 64, 65, 66, 67, 68, 115, 70, 71, 72, 73, 74, 75, 76, 87, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 129, 125, 126, 127, 128, 129, 130, 131, 132, 133]
        
        document_similarities = sorted([
            (self.vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in embeddings['embedding'].to_dict().items()
        ], reverse=True)
    
        return document_similarities
    
    def construct_prompt(self, question: str, document_similarities: dict, chunks_df: pd.DataFrame) -> str:
        """
        Fetch relevant 
        """
        MAX_SECTION_LEN = 500
        SEPARATOR = "\n* "
        ENCODING = "gpt2"  # encoding for text-davinci-003

        encoding = tiktoken.get_encoding(ENCODING)
        separator_len = len(encoding.encode(SEPARATOR))        
        chosen_sections = []
        chosen_sections_len = 0
        chosen_sections_indexes = []
        
        for _, section_index in document_similarities:
            # Add contexts until we run out of space.        
            document_section = chunks_df.loc[section_index]
            print(document_section)
            
            chosen_sections_len += document_section.tokens + separator_len
            if chosen_sections_len > MAX_SECTION_LEN:
                break
                
            chosen_sections.append(SEPARATOR + document_section.page_content.replace("\n", " "))
            chosen_sections_indexes.append(str(section_index))
                
        # Useful diagnostic information
        print(f"Selected {len(chosen_sections)} document sections:")
        print("\n".join(chosen_sections_indexes))
        
        header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
        
        return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"
    
    def answer_query_with_context(self, prompt, show_prompt: bool = False):
        COMPLETIONS_API_PARAMS = {
                    # We use temperature of 0.0 because it gives the most predictable, factual answer.
                    "temperature": 0.0,
                    "max_tokens": 300,
                    "model": COMPLETIONS_MODEL,
                     }
        if show_prompt:
            print(prompt)

        response = openai.Completion.create(
                    prompt=prompt,
                    **COMPLETIONS_API_PARAMS
                )

        return response["choices"][0]["text"].strip(" \n")
                
    def embedding_data(self, chunks):
        from langchain.vectorstores import Pinecone
        from langchain.embeddings.openai import OpenAIEmbeddings
        import pinecone

        embeddings = OpenAIEmbeddings(openai_api_key='KEY')
        pinecone.init(api_key='KEY', environment='ENV')
        index_name = 'index_name'
        vectorStore = Pinecone.from_texts([t.page_content for t in chunks], embeddings, index_name=index_name)

        return vectorStore
    
    def query_vectordatabase(vectorStore, input):
        docs = vectorStore.similarity_search(input, k=3, include_metada=True)
        logger.info(f'Total of similar docs {len(docs)}')

    def generate_docs(self, directory_path):
        documents = SimpleDirectoryReader(directory_path).load_data() #Compare versus UnstructuredPDFLoader
        return documents
    
    def query_input(query, vectorStore):
        from langchain.llms import OpenAI
        from langchain.chains.question_answering import load_qa_chain
        llm = OpenAI(temperature=0, openai_api_key='')
        chain = load_qa_chain(llm, chain_type="stuff")
        docs = vectorStore.similarity_search(query, include_metadata=True)
        chain.run(input_documents=docs, question=query)


    def construct_index(self, chunks):
        max_input_size = 4096
        num_outputs = 512
        max_chunk_overlap = 20
        chunk_size_limit = 600
        temperature=0.7
        model_name="text-davinci-003"
        index_file = 'index.json'

        prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
        llm_predictor = LLMPredictor(llm=OpenAI(temperature=temperature, model_name=model_name, max_tokens=num_outputs))
        index = GPTSimpleVectorIndex(chunks, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
        index.save_to_disk(f"vectors/{index_file}")

        return index

    def chatbot(self, input_text):
        index = GPTSimpleVectorIndex.load_from_disk('index.json')
        response = index.query(input_text, response_mode="compact")
        return response.response

    # def launch(self):
    #     documents = self.generate_docs('docs')
    #     index = self.construct_index(documents)
    #     self.iface.launch(share=True)

    def launch(self, query, file_name):
        documents = self.generate_docs_langchain(file_name)
        chunks = self.chunk_data_langchain(documents, chunk_file=file_name)
        chunks_df = self.get_df_embedded(chunks)
        document_similarities = self.obtain_similar_vectors(query, chunks_df)
        prompt = self.construct_prompt(query, document_similarities, chunks_df)
        answer = self.answer_query_with_context(query)
        print(answer)
        # vectorStore = self.construct_index(chunks)
        # self.query_vectordatabase(vectorStore, input)



if __name__ == '__main__':
    file_name = 'DeliveringHappiness'
    query = "What it means happiness?"
    gpt = GPT()
    gpt.launch(query, file_name)
