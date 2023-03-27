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