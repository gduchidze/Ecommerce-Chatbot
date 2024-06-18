import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, Vector, ServerlessSpec

pinecone = Pinecone(api_key="29fe7b06-67db-483f-87ec-906da52b6961") # for fast check, code is for [.env] 
index_name = "giorgi"  # NEW INDEX FOR EVERY RUN !!!!!
dimension = 384  # Dimension for the all-MiniLM-L6-v2 model

# Check if the index exists, otherwise create it with serverless deployment on AWS us-east-1
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name, dimension=dimension, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pinecone.Index(index_name)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    words = word_tokenize(text.lower())
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_words)


def load_products(filepath):
    products = {}
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            product_id = row['Uniq Id']  # Assuming this is the unique identifier
            description = preprocess_text(f"{row.get('About Product', '')} {row.get('Product Details', '')} "
                                            f"{row.get('Product Specification', '')} {row.get('Technical Details', '')}")
            product_info = {key.lower().replace(' ', '_'): value for key, value in row.items()}
            products[product_id] = product_info

            # Create a more comprehensive description for indexing
            description_vector = model.encode(description).tolist()
            index.upsert(vectors=[Vector(id=product_id, values=description_vector)])
    return products


def search_products(query):
    query_vector = model.encode(query).tolist()
    response = index.query(vector=Vector(values=query_vector), top_k=5, include_metadata=True)
    relevant_product_ids = [match['id'] for match in response['matches']]
    return relevant_product_ids


def get_product_details(product_id, products):
    return products.get(product_id, "Product not found.")


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
                print(f"{i+1}. Name: {product_info['product_name']}")
                print(f"   Description: {product_info['about_product']}")
                print(f"   Brand: {product_info['brand_name']}")
                print(f"   Price: {product_info['list_price']}")
                print(f"   Quantity: {product_info['quantity']}")
            print("Would you like to know more about any of these products? (Enter a number or 'no')")
            product_choice = input().lower()
            if product_choice.isdigit():
                choice_index = int(product_choice) - 1
                if 0 <= choice_index < len(relevant_product_ids):
                    chosen_product_id = relevant_product_ids[choice_index]
                    chosen_product_info = get_product_details(chosen_product_id, products)
                    print(f"Here's more information about the {chosen_product_info['product_name']}:")
                    print(f"Description: {chosen_product_info['about_product']}")
                    print(f"Technical Details: {chosen_product_info['technical_details']}")
                    print(f"Product Specification: {chosen_product_info['product_specification']}")
                else:
                    print("Invalid choice. Please try again.")
            else:
                print("No problem, happy shopping!")
        else:
            print("No products found matching your query. Try searching with different keywords.")


if __name__ == "__main__":
    main()
