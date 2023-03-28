ENVIRONMENT = 'dev'
CONTEXT = 'cxlab'
BUCKET_NAME = 'datablackgold'
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 0
COMPLETIONS_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"
COMPLETIONS_API_PARAMS = {
                    "temperature": 0.0,
                    "max_tokens": 300,
                    "model": COMPLETIONS_MODEL,
                     }
MAX_SECTION_LEN = 4097 - COMPLETIONS_API_PARAMS["max_tokens"]
SEPARATOR = "\n* "
HEADER = "Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say 'I don't know.'"