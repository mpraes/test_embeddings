import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextEmbeddingMethods:
    def __init__(self):
        self.bow_vectorizer = None
        self.tfidf_vectorizer = None
        self.word2vec_model = None
        self.glove_model = None
        
    def preprocess_text(self, text):
        """Preprocess text by tokenizing and removing stopwords"""
        # Tokenize into sentences
        sentences = sent_tokenize(text.lower())
        
        # Tokenize words and remove stopwords
        stop_words = set(stopwords.words('english'))
        processed_sentences = []
        
        for sentence in sentences:
            words = word_tokenize(sentence)
            words = [word for word in words if word.isalnum() and word not in stop_words]
            if words:
                processed_sentences.append(words)
        
        return processed_sentences
    
    def bow_embedding(self, documents):
        """
        Bag of Words (BOW) embedding
        Returns: Document-term matrix
        """
        print("=== BOW (Bag of Words) Embedding ===")
        
        # Join words back into sentences for vectorizer
        processed_docs = [' '.join(doc) for doc in documents]
        
        # Create BOW vectorizer
        self.bow_vectorizer = CountVectorizer(max_features=1000)
        bow_matrix = self.bow_vectorizer.fit_transform(processed_docs)
        
        # Convert to dense array for easier handling
        bow_array = bow_matrix.toarray()
        
        print(f"BOW Matrix Shape: {bow_array.shape}")
        print(f"Vocabulary size: {len(self.bow_vectorizer.vocabulary_)}")
        print(f"Sample features: {list(self.bow_vectorizer.vocabulary_.keys())[:10]}")
        
        return bow_array
    
    def tfidf_embedding(self, documents):
        """
        TF-IDF embedding
        Returns: TF-IDF matrix
        """
        print("\n=== TF-IDF Embedding ===")
        
        # Join words back into sentences for vectorizer
        processed_docs = [' '.join(doc) for doc in documents]
        
        # Create TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_docs)
        
        # Convert to dense array
        tfidf_array = tfidf_matrix.toarray()
        
        print(f"TF-IDF Matrix Shape: {tfidf_array.shape}")
        print(f"Vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
        print(f"Sample features: {list(self.tfidf_vectorizer.vocabulary_.keys())[:10]}")
        
        return tfidf_array
    
    def word2vec_embedding(self, documents, vector_size=100, window=5, min_count=1):
        """
        Word2Vec embedding
        Returns: Document embeddings (average of word vectors)
        """
        print("\n=== Word2Vec Embedding ===")
        
        # Train Word2Vec model
        self.word2vec_model = Word2Vec(
            sentences=documents,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
            sg=1  # Skip-gram
        )
        
        # Get document embeddings by averaging word vectors
        doc_embeddings = []
        for doc in documents:
            word_vectors = []
            for word in doc:
                if word in self.word2vec_model.wv:
                    word_vectors.append(self.word2vec_model.wv[word])
            
            if word_vectors:
                doc_embedding = np.mean(word_vectors, axis=0)
            else:
                doc_embedding = np.zeros(vector_size)
            
            doc_embeddings.append(doc_embedding)
        
        doc_embeddings = np.array(doc_embeddings)
        
        print(f"Word2Vec Document Embeddings Shape: {doc_embeddings.shape}")
        print(f"Vocabulary size: {len(self.word2vec_model.wv.key_to_index)}")
        print(f"Vector size: {vector_size}")
        
        return doc_embeddings
    
    def glove_embedding(self, documents, vector_size=100):
        """
        GloVe embedding (using pre-trained model simulation)
        Returns: Document embeddings (average of word vectors)
        """
        print("\n=== GloVe Embedding ===")
        
        # For demonstration, we'll train a Word2Vec model and treat it as GloVe
        # In practice, you would load pre-trained GloVe vectors
        self.glove_model = Word2Vec(
            sentences=documents,
            vector_size=vector_size,
            window=10,  # GloVe typically uses larger windows
            min_count=1,
            workers=4,
            sg=0  # CBOW (GloVe-like)
        )
        
        # Get document embeddings by averaging word vectors
        doc_embeddings = []
        for doc in documents:
            word_vectors = []
            for word in doc:
                if word in self.glove_model.wv:
                    word_vectors.append(self.glove_model.wv[word])
            
            if word_vectors:
                doc_embedding = np.mean(word_vectors, axis=0)
            else:
                doc_embedding = np.zeros(vector_size)
            
            doc_embeddings.append(doc_embedding)
        
        doc_embeddings = np.array(doc_embeddings)
        
        print(f"GloVe Document Embeddings Shape: {doc_embeddings.shape}")
        print(f"Vocabulary size: {len(self.glove_model.wv.key_to_index)}")
        print(f"Vector size: {vector_size}")
        
        return doc_embeddings
    
    def compare_embeddings(self, bow_emb, tfidf_emb, w2v_emb, glove_emb):
        """Compare different embedding methods"""
        print("\n=== Embedding Comparison ===")
        
        # Calculate sparsity for BOW and TF-IDF
        bow_sparsity = 1.0 - np.count_nonzero(bow_emb) / bow_emb.size
        tfidf_sparsity = 1.0 - np.count_nonzero(tfidf_emb) / tfidf_emb.size
        
        print(f"BOW Sparsity: {bow_sparsity:.3f}")
        print(f"TF-IDF Sparsity: {tfidf_sparsity:.3f}")
        print(f"Word2Vec Dense: {w2v_emb.shape}")
        print(f"GloVe Dense: {glove_emb.shape}")
        
        # Calculate document similarities (cosine similarity)
        from sklearn.metrics.pairwise import cosine_similarity
        
        print("\nDocument Similarities (Cosine):")
        print("BOW Similarity Matrix:")
        bow_sim = cosine_similarity(bow_emb)
        print(bow_sim)
        
        print("\nTF-IDF Similarity Matrix:")
        tfidf_sim = cosine_similarity(tfidf_emb)
        print(tfidf_sim)
        
        print("\nWord2Vec Similarity Matrix:")
        w2v_sim = cosine_similarity(w2v_emb)
        print(w2v_sim)
        
        print("\nGloVe Similarity Matrix:")
        glove_sim = cosine_similarity(glove_emb)
        print(glove_sim)
    
    def visualize_embeddings(self, embeddings, method_name):
        """Visualize embeddings using PCA"""
        print(f"\n=== {method_name} Visualization ===")
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
        plt.title(f'{method_name} Embeddings (PCA)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        
        # Add document labels
        for i, (x, y) in enumerate(embeddings_2d):
            plt.annotate(f'Doc {i+1}', (x, y), xytext=(5, 5), textcoords='offset points')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{method_name}_embeddings.png')
        plt.close()

def main():
    # Sample text document
    sample_text = """
    Artificial intelligence is transforming the way we live and work. Machine learning algorithms 
    can now process vast amounts of data to identify patterns and make predictions. Deep learning 
    models, particularly neural networks, have achieved remarkable success in image recognition, 
    natural language processing, and speech recognition tasks.
    
    Natural language processing enables computers to understand and generate human language. 
    This technology powers virtual assistants, translation services, and sentiment analysis tools. 
    The field continues to advance rapidly with new architectures and training methods.
    
    Computer vision allows machines to interpret and analyze visual information from the world. 
    Applications include autonomous vehicles, medical imaging, and facial recognition systems. 
    These technologies are becoming increasingly sophisticated and accurate.
    """
    
    # Save sample text to file
    with open('sample_text.txt', 'w') as f:
        f.write(sample_text)
    
    print("Sample Document:")
    print(sample_text)
    print("=" * 80)
    
    # Initialize embedding methods
    embedder = TextEmbeddingMethods()
    
    # Preprocess the text
    processed_docs = embedder.preprocess_text(sample_text)
    
    # Save processed docs
    with open('processed_docs.txt', 'w') as f:
        f.write(f"Processed into {len(processed_docs)} sentences\n")
        f.write("Sample processed sentences:\n")
        for i, doc in enumerate(processed_docs[:3]):
            f.write(f"  {i+1}: {' '.join(doc)}\n")
    
    print(f"Processed into {len(processed_docs)} sentences")
    print("Sample processed sentences:")
    for i, doc in enumerate(processed_docs[:3]):
        print(f"  {i+1}: {doc}")
    
    # Apply different embedding methods
    bow_embeddings = embedder.bow_embedding(processed_docs)
    tfidf_embeddings = embedder.tfidf_embedding(processed_docs)
    word2vec_embeddings = embedder.word2vec_embedding(processed_docs)
    glove_embeddings = embedder.glove_embedding(processed_docs)
    
    # Save embeddings to files
    np.savetxt('bow_embeddings.txt', bow_embeddings)
    np.savetxt('tfidf_embeddings.txt', tfidf_embeddings)
    np.savetxt('word2vec_embeddings.txt', word2vec_embeddings)
    np.savetxt('glove_embeddings.txt', glove_embeddings)
    
    # Compare embeddings
    embedder.compare_embeddings(bow_embeddings, tfidf_embeddings, 
                               word2vec_embeddings, glove_embeddings)
    
    # Visualize embeddings (now saves to files)
    embedder.visualize_embeddings(bow_embeddings, "BOW")
    embedder.visualize_embeddings(tfidf_embeddings, "TF-IDF")
    embedder.visualize_embeddings(word2vec_embeddings, "Word2Vec")
    embedder.visualize_embeddings(glove_embeddings, "GloVe")
    
    # Example usage: Find similar words using Word2Vec
    with open('similar_words.txt', 'w') as f:
        f.write("=== Word2Vec Similar Words ===\n")
        if 'artificial' in embedder.word2vec_model.wv:
            similar_words = embedder.word2vec_model.wv.most_similar('artificial', topn=5)
            f.write(f"Words similar to 'artificial': {similar_words}\n")
        
        if 'learning' in embedder.word2vec_model.wv:
            similar_words = embedder.word2vec_model.wv.most_similar('learning', topn=5)
            f.write(f"Words similar to 'learning': {similar_words}\n")

if __name__ == "__main__":
    main()
