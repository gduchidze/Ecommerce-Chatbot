import csv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, Vector, ServerlessSpec

#The code downloads some necessary data for language processing, 
#like stopwords (common words like "and", "the") and rules for changing words to their 
#root form (lemmatization).

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Pinecone API key
pinecone_api_key = "68fe7b22-8afc-4ca0-8e10-40e069a1d2bb"

# Initialize Pinecone
pinecone = Pinecone(api_key=pinecone_api_key)
index_name = "new"
dimension = 384  # Dimension for the all-MiniLM-L6-v2 model

if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name, dimension=dimension, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

#We load a smart model called all-MiniLM-L6-v2 which helps us turn sentences into numbers.

index = pinecone.Index(index_name)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

#We create a function to clean up text by removing common words and changing words to their simplest form.
def preprocess_text(text):
    words = word_tokenize(text.lower())
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_words)

# This function reads product data from a CSV file, cleans up the text descriptions, 
# and stores them in Pinecone. It also keeps a local copy of the product information.
def load_products(filepath):
    products = {}
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            product_id = row['Uniq Id']  # Assuming this is the unique identifier
            description = preprocess_text(f"{row.get('About Product', '')} {row.get('Product Details', '')} "
                                          f"{row.get('Product Specification', '')} {row.get('Technical Details', '')}")
            product_info = {
                'name': row['Product Name'],
                'brand_name': row.get('Brand Name', ''),  
                'asin': row.get('Asin', ''),
                'category': row.get('Category', ''),
                'upc_ean_code': row.get('Upc Ean Code', ''),
                'list_price': row.get('List Price', ''),
                'selling_price': row.get('Selling Price', ''),
                'quantity': row.get('Quantity', ''),
                'model_number': row.get('Model Number', ''),
                'dimensions': row.get('Dimensions', ''), 
                'color': row.get('Color', ''),
                'ingredients': row.get('Ingredients', ''),
                'direction_to_use': row.get('Direction To Use', ''),
                'is_amazon_seller': row.get('Is Amazon Seller', ''),
                'image': row.get('Image', ''),  
                'variants': row.get('Variants', ''),
                'sku': row.get('Sku', ''),
                'product_url': row.get('Product Url', ''),  
                'stock': row.get('Stock', ''),
                'description': description  
            }
            products[product_id] = product_info

            description_vector = model.encode(description).tolist()
            index.upsert(vectors=[Vector(id=product_id, values=description_vector)])
    return products

#This function takes a user's search query,
#turns it into numbers, and searches Pinecone for the most similar product descriptions.

def search_products(query):
    query_vector = model.encode(query).tolist()
    response = index.query(vector=Vector(id='query', values=query_vector), top_k=5, include_metadata=True)
    relevant_product_ids = [match['id'] for match in response['matches']]
    return relevant_product_ids


def get_product_details(product_id, products):
    return products.get(product_id, "Product not found.")

#It loads the products, and then enters a loop where
#it asks the user what they are looking for, searches for matching products, and displays the results.

def main():
    print("Hi! I'm Giorgi, your friendly Amazon e-commerce support assistant. How can I help you today?")
    products = load_products("data/sample_data_10k.csv")
    while True:
        query = input("What are you looking for? ").lower()
        if query == "exit":
            break

        relevant_product_ids = search_products(query)
        if relevant_product_ids:
            print("Here are some products that might interest you:")
            for i, product_id in enumerate(relevant_product_ids):
                product_info = get_product_details(product_id, products)
                print(f"{i+1}. Name: {product_info['name']}")
            print("Would you like to know more about any of these products? (Enter a number or 'no')")
            product_choice = input().lower()
            if product_choice.isdigit():
                choice_index = int(product_choice) - 1
                if 0 <= choice_index < len(relevant_product_ids):
                    chosen_product_id = relevant_product_ids[choice_index]
                    chosen_product_info = get_product_details(chosen_product_id, products)
                    print(f"Here's more information about the {chosen_product_info['name']}:")
                    print(f"Description: {chosen_product_info['description']}")
                else:
                    print("Invalid choice. Please try again.")
            else:
                print("No problem, happy shopping!")
        else:
            print("No products found matching your query. Try searching with different keywords.")


if __name__ == "__main__":
    main()
