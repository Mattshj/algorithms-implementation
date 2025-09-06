"""
Unified Algorithms Module
Author: Matthew Jaberi
Description: Implementation of K-Means clustering, Image Compression using K-Means,
and a Naive Bayes text classifier. Optimized for readability, maintainability, and performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte
from matplotlib import image as mp_image
from collections import defaultdict
from nltk.tokenize import word_tokenize


# -------------------------
# K-MEANS CLUSTERING
# -------------------------
class KMeans:
    """K-Means clustering implementation (vectorized where possible)."""

    def __init__(self, data: np.ndarray, k: int):
        self.data = data
        self.k = k
        self.centers = self.data[np.random.choice(data.shape[0], k, replace=False)]
        self.clusters = np.zeros(data.shape[0], dtype=int)

    def fit(self, iterations: int):
        """Fit the model for a given number of iterations using efficient numpy operations."""
        for _ in range(iterations):
            # Compute distances vectorized
            distances = np.linalg.norm(self.data[:, np.newaxis] - self.centers, axis=2)
            self.clusters = np.argmin(distances, axis=1)

            # Update centroids
            for i in range(self.k):
                if np.any(self.clusters == i):
                    self.centers[i] = np.mean(self.data[self.clusters == i], axis=0)
        return self.centers, self.clusters


# -------------------------
# IMAGE COMPRESSION USING K-MEANS
# -------------------------
class ImageCompressor:
    """Compress images using K-Means clustering."""

    def __init__(self, image_path: str, n_colors: int = 256):
        self.image_path = image_path
        self.n_colors = n_colors
        self.img = mp_image.imread(image_path)
        self.compressed_image = None

    def compress(self, n_iterations: int = 20):
        """Compress the image using K-Means and return the result as a NumPy array."""
        img_flat = self.img.reshape(-1, 3)
        kmeans = KMeans(img_flat, self.n_colors)
        centroids, labels = kmeans.fit(n_iterations)

        self.compressed_image = centroids[labels].reshape(self.img.shape)
        return self.compressed_image

    def save_and_show(self, output_path: str):
        """Convert to uint8, save, and display the compressed image."""
        img_uint8 = img_as_ubyte(self.compressed_image, force_copy=False)
        plt.imshow(img_uint8)
        plt.title(f'Image Compressed to {self.n_colors} Colors')
        plt.axis('off')
        plt.show()
        io.imsave(output_path, img_uint8)


# -------------------------
# NAIVE BAYES TEXT CLASSIFIER
# -------------------------
class NaiveBayes:
    """Naive Bayes text classifier with stopwords removal and vectorized probability calculation."""

    def __init__(self, alpha: float = 1.0, stopwords_path: str = "sw.txt"):
        self.alpha = alpha
        self.stopwords_path = stopwords_path
        self.classes = None
        self.dic = None
        self.p_c = None
        self.vocab = None
        self.vocab_length = 0
        self.d = None

    def _load_stopwords(self):
        with open(self.stopwords_path, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f.readlines())

    def fit(self, train_data, labels):
        """Fit the Naive Bayes model to training data."""
        self.classes = np.unique(labels)
        self.dic = np.array([defaultdict(lambda: 0) for _ in self.classes])
        stopwords = self._load_stopwords()

        for idx, class_val in enumerate(self.classes):
            class_texts = [train_data[i] for i in range(len(labels)) if labels[i] == class_val]
            for text in class_texts:
                tokens = [w for w in word_tokenize(text) if w not in stopwords]
                for token in tokens:
                    self.dic[idx][token] += 1

        # Class probabilities
        self.p_c = np.array([np.sum(labels == class_val) / len(labels) for class_val in self.classes])

        # Vocabulary and denominators for smoothing
        all_words = []
        cat_word_counts = np.zeros(len(self.classes))
        for idx in range(len(self.classes)):
            cat_word_counts[idx] = sum(self.dic[idx].values())
            all_words += self.dic[idx].keys()

        self.vocab = np.unique(np.array(all_words))
        self.vocab_length = len(self.vocab)
        self.d = cat_word_counts + self.vocab_length + self.alpha

    def predict(self, test_data):
        """Predict classes for the test dataset."""
        stopwords = self._load_stopwords()
        predictions = []

        for text in test_data:
            tokens = [w for w in word_tokenize(text) if w not in stopwords]
            log_probs = np.zeros(len(self.classes))
            for idx in range(len(self.classes)):
                token_probs = np.array([
                    (self.dic[idx].get(token, 0) + self.alpha) / self.d[idx] for token in tokens
                ])
                log_probs[idx] = np.sum(np.log(token_probs)) + np.log(self.p_c[idx])
            predictions.append(self.classes[np.argmax(log_probs)])
        return np.array(predictions)


# -------------------------
# SCRIPT EXAMPLES
# -------------------------
if __name__ == "__main__":
    # Example: Image Compression
    compressor = ImageCompressor("image.png")
    compressor.compress()
    compressor.save_and_show("image_compressed_K256.png")

    # Example: Naive Bayes (pseudo code)
    # train_texts = ["sample text 1", "sample text 2"]
    # labels = np.array(["class1", "class2"])
    # nb = NaiveBayes(alpha=1.0)
    # nb.fit(train_texts, labels)
    # predictions = nb.predict(["new text"])
