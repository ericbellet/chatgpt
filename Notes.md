Notes:

The steps are:

Preprocess the contextual information by splitting it into chunks and create an embedding vector for each chunk.
On receiving a query, embed the query in the same vector space as the context chunks and find the context embeddings which are most similar to the query.
Prepend the most relevant context embeddings to the query prompt.
Submit the question along with the most relevant context to GPT, and receive an answer which makes use of the provided contextual information.
https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb

chunks = textwrap.wrap(article,4000)

Salience (que herramienta usar para una tarea) and Anticipation. 

FIne-tunning. 

Retrieval-augmented generation (https://towardsdatascience.com/generative-question-answering-with-long-term-memory-c280e237b144)

We’re ready to combine OpenAI’s Completion and Embedding endpoints with our Pinecone vector database to create a retrieval-augmented GQA system.

Memory is the concept of persisting state between calls of a chain/agent. LangChain provides a standard interface for memory, a collection of memory implementations, and examples of chains/agents that use memory.

https://github.com/jamescalam/openai-cookbook/blob/main/techniques_to_improve_reliability.md

https://www.pinecone.io/learn/nlp/

Chunk data: https://gist.github.com/jamescalam/97c9e72de78c51c59a9b915186df7733#file-openai-transcripts-merge-snippets-ipynb
Chunk in paralel: https://github.com/pinecone-io/examples/blob/master/search/semantic-search/jeopardy/jeopardy.ipynb
Deduplication: https://github.com/pinecone-io/examples/blob/master/search/semantic-search/deduplication/deduplication_scholarly_articles.ipynb
Preprocess data: https://github.com/gkamradt/langchain-tutorials/blob/main/data_generation/Clean%20and%20Standardize%20Data.ipynb
Dibujo de arquitectura: https://betterprogramming.pub/fixing-youtube-search-with-openais-whisper-90bb569073cf
Overlab chunks: https://gist.github.com/jamescalam/17df40133d11c3c25aa9f4045c9d1145#file-whisper-yt-search-longer-segments-ipynb
Long texts: https://github.com/jamescalam/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
Recommend embbedings and cache: https://github.com/jamescalam/openai-cookbook/blob/main/examples/Recommendation_using_embeddings.ipynb
Semantic Search using embeddings: https://github.com/jamescalam/openai-cookbook/blob/main/examples/Semantic_text_search_using_embeddings.ipynb
Long texts: https://github.com/jamescalam/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
Advance propmpts: https://github.com/jamescalam/openai-cookbook/blob/main/techniques_to_improve_reliability.md
Improve fine-tunning: https://docs.google.com/document/d/1rqj7dkuvl7Byd5KQPUJRxc19BJt8wo0yHNwK84KfU3Q/edit
Memory: https://langchain.readthedocs.io/en/latest/modules/memory/getting_started.html (https://www.youtube.com/watch?v=7RcS8FHGdW4&list=WL&index=1)

OpenAI Embedding endpoint to create vector representations of each query.
Pinecone vector database to search for relevant passages from the database of previously indexed contexts.
OpenAI Completion endpoint to generate a natural language answer considering the retrieved contexts.

The “text-davinci-003” model instead of the latest “gpt-3.5-turbo” model because Davinci works much better for text completion. If you want, you can very well change the model to Turbo to reduce the cost.

Al ofrecer tus servicios como fullstack remoto, es importante ser claro y transparente con los precios que cobras a tus clientes. Aquí hay un ejemplo de propuesta que podrías usar para presentar tus precios de manera clara y efectiva:

Estimado [nombre del cliente],

Gracias por considerar mis servicios como fullstack remoto para su proyecto [nombre del proyecto]. He revisado los requisitos del proyecto y estoy emocionado de ofrecerle mis servicios. A continuación, encontrará una descripción detallada de mis servicios y los precios correspondientes.

Descripción de los servicios:

Diseño y desarrollo de la aplicación web (front-end y back-end)
Configuración de servidor y alojamiento web
Integración de bases de datos y herramientas de terceros
Pruebas de calidad y corrección de errores
Mantenimiento continuo y soporte técnico
Precios:

Tarifa por hora: $XX/hora
Tarifa por proyecto: $XXXX
Nota: Los precios pueden variar según la complejidad del proyecto y la cantidad de horas requeridas para completarlo. Por favor, póngase en contacto conmigo para discutir más detalles y recibir un presupuesto personalizado.

Espero que esta propuesta sea de su agrado y me permita trabajar juntos en su proyecto. Por favor, no dude en ponerse en contacto conmigo si tiene alguna pregunta o necesita más información.

Atentamente,
[Tu nombre]


INFO:gpt_index.token_counter.token_counter:> [query] Total LLM token usage: 758 tokens
INFO:gpt_index.token_counter.token_counter:> [query] Total embedding token usage: 9 tokens
INFO:gpt_index.token_counter.token_counter:> [query] Total LLM token usage: 690 tokens
INFO:gpt_index.token_counter.token_counter:> [query] Total embedding token usage: 9 tokens


Breaking our knowledge database on small chunks and indexing these small chunks.

GPT Index: Index data

Prompt helper.

    This utility helps us fill in the prompt, split the text,
    and fill in context information according to necessary token limitations.

The GPTSimpleVectorIndex is a data structure where nodes are keyed by
    embeddings, and those embeddings are stored within a simple dictionary.
    During index construction, the document texts are chunked up,
    converted to nodes with text; they are then encoded in
    document embeddings stored within the dict.

    During query time, the index uses the dict to query for the top
    k most similar nodes, and synthesizes an answer from the
    retrieved nodes.

     GPTSimpleVectorIndex es una librería de Python que se utiliza para construir un índice de vectores simples.

     Un índice de vectores es una técnica de recuperación de información que se utiliza para acelerar la búsqueda de información en grandes conjuntos de datos. En lugar de buscar en todo el conjunto de datos, un índice de vectores utiliza una representación de baja dimensionalidad de los datos para buscar información. Los índices de vectores se utilizan a menudo en la recuperación de información de texto, la recuperación de imágenes y la recuperación de información de audio.


There are different strategies to convert long text data into something more optimized for LLMs, depending on the use case and the desired output. Here are some possible strategies:

Truncating the input text

Summarization: This strategy involves reducing the number of words by extracting or generating the most important information from the long text data. Summarization can help LLMs focus on the main points and avoid irrelevant details. However, summarization may also lose some nuances or context that could be useful for certain tasks1.
Chunking: This strategy involves splitting the long text data into smaller segments or chunks that can fit into the LLM’s input size. Chunking can help LLMs process long texts more efficiently and avoid truncation errors. However, chunking may also introduce discontinuities or inconsistencies between chunks that could affect the quality of the output1.
Labeling: This strategy involves adding metadata or annotations to the long text data that can provide additional information or guidance for the LLM. Labeling can help LLMs understand the structure, format, or domain of the long text data and perform more specific tasks. However, labeling may also require manual effort or domain expertise that could be costly or impractical1.
Transformation: This strategy involves changing the representation or format of the long text data to make it more compatible with the LLM’s architecture or objective function. Transformation can help LLMs leverage their pre-trained knowledge and adapt to new tasks more easily. However, transformation may also introduce noise or distortion that could degrade the performance of the LLM1.
These strategies are not mutually exclusive and can be combined in different ways to optimize long text data for LLMs2. The choice of strategy depends on factors such as:

The type and size of long text data
The type and size of LLM
The task and output format
The available resources and constraints


For chunking, some possible strategies are:

Using short paragraphs and text lines with white space to separate them.
Creating clear visual hierarchies with related items grouped together
Using distinct groupings in strings of letters or numbers such as passwords, license keys, credit-card or account numbers, phone numbers, and dates1
Using sticky notes to mark natural stopping points in the text2
Using a file folder to help focus on one section of the text at a time2

For summarization, some possible strategies are:

Using mnemonics, acronyms, acrostics, and other strategies as ways to chunk different units of information into a memorable word or phrase3
Using bullet points or lists to highlight the main ideas or key points of the text
Using headings or subheadings to organize the text into meaningful sections
Writing a short summary sentence or paragraph at the end of each section or chunk.

Some possible examples of chunking strategies for a book such as Delivering Happiness are:

Dividing the book into chapters based on the main topics or themes
Dividing each chapter into sections based on the subtopics or key points
Dividing each section into paragraphs based on the supporting details or examples
Using headings, subheadings, bullet points, and white space to separate and highlight the chunks
Using sticky notes or bookmarks to mark the chunks and write summaries or questions

Some possible examples of summarization strategies for a book such as Delivering Happiness are:

Writing a short summary sentence or paragraph at the end of each chapter or section that captures the main idea and key points
Writing a short summary sentence or paragraph at the beginning of the book that introduces the purpose and scope of the book
Writing a short summary sentence or paragraph at the end of the book that concludes the main message and takeaways of the book
Using acronyms, acrostics, or mnemonics to remember the main points or themes of the book

Some possible examples of labeling strategies for a book such as Delivering Happiness are:

Using title element to label the HTML document that contains the book
Using alt attribute to label the image of the book cover
Using headings or subheadings to label the chapters and sections of the book
Using optgroup element to label the groups of chapters or sections in a table of contents
Using aria-label or aria-labelledby attributes to label any dialogs or interactive elements in the book.

UnstructuredPDFLoader can provide more information about the source and structure of the documents, while SimpleDirectoryReader can provide faster and simpler access to the documents.