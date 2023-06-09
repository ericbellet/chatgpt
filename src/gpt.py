import configparser
import pandas as pd
import tiktoken
import tempfile
import pickle
import openai
import os
import io
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai.embeddings_utils import get_embedding, cosine_similarity
from utils import logger
from parameters import *
from s3 import S3


def num_tokens_from_string(string: str, encoding_name="cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def load_tempfile(data):
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = f"{temp_dir}/online_file.pdf"
        file = open(file_path, "wb")
        file.write(data)
        file.close()
        loader = UnstructuredPDFLoader(file_path)
        return loader.load()
 
class GPT():
    
    def __init__(self) -> None:
        config_file = 'config.ini'
        with open(config_file) as f:
            config = configparser.ConfigParser()
            config.read_file(f)
        os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY') or config.get(CONTEXT, 'OPENAI_API_KEY')
        openai.api_key = os.environ["OPENAI_API_KEY"]
        if ENVIRONMENT != 'local':
            self.s3 = S3()
    
    def generate_docs(self, file):
        data = []
        s3_file = self.s3.check_file_exists(BUCKET_NAME, f'chunks/{file}.pkl') if ENVIRONMENT != 'local' else False

        if ENVIRONMENT == 'local' and not os.path.exists(f"chunks/{file}.pkl"):
            loader = UnstructuredPDFLoader(f'docs/{file}.pdf')
            data = loader.load()
            
        elif ENVIRONMENT != 'local' and not s3_file:
            pdf_file = self.s3.download_file(bucket_name=BUCKET_NAME, obj=f'docs/{file}.pdf')
            data = load_tempfile(pdf_file.getvalue())

        logger.info(f'You have {len(data)} document(s) in your data')        
        return data

    def chunk_data(self, data, chunk_file, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        file = f'chunks/{chunk_file}.pkl'
        s3_file = self.s3.check_file_exists(BUCKET_NAME, file) if ENVIRONMENT != 'local' else False

        if ENVIRONMENT == 'local' and os.path.exists(file):
            with open(file, 'rb') as f:
                chunks = pickle.load(f)
        elif ENVIRONMENT != 'local' and s3_file:
            chunks = self.s3.download_file(bucket_name=BUCKET_NAME, obj=file)
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = text_splitter.split_documents(data)
            logger.info(f'Now you have {len(chunks)} documents')
            if ENVIRONMENT == 'local':
                with open(f'chunks/{chunk_file}.pkl', 'wb') as f:
                    pickle.dump(chunks, f)
            else:
                self.s3.upload_file(bucket=BUCKET_NAME, data=pickle.dumps(chunks), object_name=file)

        return chunks
    
    def obtain_embedding(self, chunk_number, page_content, engine):
        logger.info(f"Chunk number: {chunk_number}")
        return get_embedding(page_content, engine)

    def get_df_embedded(self, chunks, vector_name):
        logger.info(f'Embedding total of chunks: {len(chunks)}')
        file = f'vectors/{vector_name}.pkl'
        s3_file = self.s3.check_file_exists(BUCKET_NAME, file) if ENVIRONMENT != 'local' else True

        if (not os.path.exists(file) and ENVIRONMENT == 'local') or (not s3_file):
            df = pd.DataFrame([{'page_content':chunk.page_content, 
                                'tokens': num_tokens_from_string(chunk.page_content),
                                'embedding': self.obtain_embedding(chunk_number, chunk.page_content, engine=EMBEDDING_MODEL)} 
                                                                                for chunk_number, chunk in enumerate(chunks)])
            if ENVIRONMENT == 'local':
                df.to_pickle(f'vectors/{vector_name}.pkl')
            else:
                self.s3.upload_file(BUCKET_NAME, pickle.dumps(df), f'vectors/{vector_name}.pkl')

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
        logger.info(f"Maximum prompt's tokens: {max_section_len}")
        
        for _, section_index in document_similarities:
            document_section = vectors_df.loc[section_index]            
            chosen_sections_len += document_section.tokens + extra_tokens
            if chosen_sections_len > max_section_len:
                break
                
            chosen_sections.append(SEPARATOR + document_section.page_content.replace("\n", " "))
            chosen_sections_indexes.append(str(section_index))
                
        logger.info(f"Selected {len(chosen_sections)} document sections.")
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
    
    def eval_same_language(self, question, answer):
        # from langdetect import detect
        # question_lang = detect(question)
        # answer_lang = detect(answer)
        import langid
        langid.set_languages(['es', 'en', 'ca'])
        question_lang, score = langid.classify(question)
        answer_lang, score = langid.classify(answer)


        #question_lang = 'es' if question_lang not in ['ca', 'es', 'en', 'fr'] else question_lang

        if question_lang == answer_lang:
            return (True, question_lang)
        else:
            return (False, question_lang)

    def translate_answer(self, answer, language):
        logger.info(f"Translating answer {answer}")
        prompt = f"Translate to {language} this: {answer}"
        return self.answer_query_with_context(prompt) 

    def load_vectors(self, vectors, path):
        dfs = []
        for vector in vectors:
            if ENVIRONMENT == 'local':
                file = os.path.join(path, vector)
            else:
                vector = vector['Key']
                file = self.s3.download_file(bucket_name=BUCKET_NAME, obj=vector)
                
            if vector.endswith('.pkl'):
                logger.info(f'Loading vector: {vector}')
                df = pd.read_pickle(file)
                dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True)

    def load_vectorDB(self, vectors=[], path='./vectors'):
        logger.info(f"Loading vector database of the environment: {ENVIRONMENT}")
        vectors = os.listdir(path) if ENVIRONMENT == 'local' else self.s3.list_objects(bucket_name=BUCKET_NAME, 
                                                                                       prefix='vectors/')['Contents']
        self.vectorDB = self.load_vectors(vectors, path)

    def preprocess(self, file_name):
        documents = self.generate_docs(file_name)
        chunks = self.chunk_data(documents, chunk_file=file_name)
        self.get_df_embedded(chunks, vector_name=file_name)

    def query(self, query):
        document_similarities = self.obtain_similar_vectors(query, self.vectorDB)
        prompt = self.construct_prompt(query, document_similarities, self.vectorDB)
        answer = self.answer_query_with_context(prompt)
        same_language = self.eval_same_language(query, answer)
        if same_language[0] == True:
            return answer
        else:
            return self.translate_answer(answer=answer, language=same_language[1]) 

if __name__ == '__main__':
    file_names = ['DeliveringHappiness', 
                  'Handbook-ExperienceEconomyPastPresentandFuture',
                  'Pine_Gilmore_The_experience_economy_1999']
    preprocessData = GPT()
    # [preprocessData.preprocess(file_name) for file_name in file_names]
    preprocessData.load_vectorDB()
    print(preprocessData.query("Dame estrategias"))
