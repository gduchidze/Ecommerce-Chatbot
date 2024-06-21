Data Preprocessing:

The script uses the NLTK library to download stopwords, which are common words like "and" or "the" that are removed from text because they don't add much meaning.
It also downloads the WordNet database for lemmatization, which is the process of reducing words to their base or root form (e.g., "running" to "run").
Pinecone Setup:

Pinecone is a vector database service used for similarity search. The script initializes a Pinecone index to store and search product descriptions represented as vectors.
Text Embedding:

The script uses the Sentence Transformers library to load a pre-trained language model (all-MiniLM-L6-v2) that converts sentences into fixed-length numerical vectors. This is done to represent product descriptions as vectors for efficient similarity search.
Data Loading:

The script reads product data from a CSV file and preprocesses the text by combining relevant fields (About Product, Product Details, Product Specification, Technical Details), cleaning the text (removing stopwords, lemmatizing), and storing the product information and its vector representation in the Pinecone index.
Search Functionality:

When a user enters a search query, the script converts the query into a vector using the same language model.
It then queries the Pinecone index to find the most similar product descriptions to the query vector.
The top matching products are retrieved and displayed to the user.
User Interaction:

The script continuously prompts the user for search queries and displays relevant product information based on the user's input.
