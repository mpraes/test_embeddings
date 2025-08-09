# Text Embedding Methods Comparison

This project demonstrates four different text embedding methods using a sample document about artificial intelligence and machine learning.

## Methods Implemented

### 1. **BOW (Bag of Words)**
- **Description**: Counts the frequency of each word in the document
- **Characteristics**: 
  - Simple and interpretable
  - High-dimensional and sparse
  - Ignores word order and context
- **Use Cases**: Basic text classification, document similarity

### 2. **TF-IDF (Term Frequency-Inverse Document Frequency)**
- **Description**: Weights words by their frequency in the document and rarity across all documents
- **Characteristics**:
  - Reduces importance of common words
  - Better than BOW for document similarity
  - Still sparse and high-dimensional
- **Use Cases**: Information retrieval, document ranking

### 3. **Word2Vec**
- **Description**: Neural network-based method that learns word embeddings from context
- **Characteristics**:
  - Dense, low-dimensional vectors
  - Captures semantic relationships
  - Can find similar words
- **Use Cases**: Word similarity, semantic analysis, feature engineering

### 4. **GloVe (Global Vectors for Word Representation)**
- **Description**: Global matrix factorization method for word embeddings
- **Characteristics**:
  - Combines global statistics with local context
  - Often performs better than Word2Vec
  - Pre-trained models available
- **Use Cases**: Advanced NLP tasks, semantic similarity

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the main script:
```bash
python main.py
```

## Code Structure

### `TextEmbeddingMethods` Class

The main class contains methods for each embedding technique:

- `preprocess_text()`: Tokenizes and cleans the input text
- `bow_embedding()`: Implements Bag of Words
- `tfidf_embedding()`: Implements TF-IDF
- `word2vec_embedding()`: Implements Word2Vec
- `glove_embedding()`: Implements GloVe (simulated)
- `compare_embeddings()`: Compares different methods
- `visualize_embeddings()`: Creates PCA visualizations

### Sample Document

The example uses a text about AI and machine learning with three paragraphs covering:
- Artificial intelligence and machine learning
- Natural language processing
- Computer vision

## Output

The script provides:

1. **Embedding Statistics**: Matrix shapes, vocabulary sizes, sparsity
2. **Similarity Matrices**: Cosine similarity between documents for each method
3. **Visualizations**: PCA plots showing document relationships
4. **Word Similarities**: Example of finding similar words using Word2Vec

## Key Differences

| Method | Dimensionality | Sparsity | Semantic Understanding | Training Time |
|--------|---------------|----------|----------------------|---------------|
| BOW | High | Very Sparse | None | Fast |
| TF-IDF | High | Sparse | None | Fast |
| Word2Vec | Low | Dense | Good | Medium |
| GloVe | Low | Dense | Excellent | Slow |

## Usage Examples

### Basic Usage
```python
from main import TextEmbeddingMethods

# Initialize
embedder = TextEmbeddingMethods()

# Process text
text = "Your text here..."
processed_docs = embedder.preprocess_text(text)

# Get embeddings
bow_emb = embedder.bow_embedding(processed_docs)
tfidf_emb = embedder.tfidf_embedding(processed_docs)
w2v_emb = embedder.word2vec_embedding(processed_docs)
glove_emb = embedder.glove_embedding(processed_docs)
```

### Custom Parameters
```python
# Word2Vec with custom parameters
w2v_emb = embedder.word2vec_embedding(
    processed_docs,
    vector_size=200,  # Embedding dimension
    window=10,        # Context window size
    min_count=2       # Minimum word frequency
)
```

## Notes

- The GloVe implementation uses Word2Vec with CBOW as a simulation
- For production use, consider using pre-trained GloVe vectors
- The visualizations help understand document relationships
- All methods are compared using cosine similarity

## Dependencies

- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `scikit-learn`: BOW, TF-IDF, and similarity metrics
- `gensim`: Word2Vec implementation
- `nltk`: Text preprocessing
- `matplotlib` & `seaborn`: Visualizations
