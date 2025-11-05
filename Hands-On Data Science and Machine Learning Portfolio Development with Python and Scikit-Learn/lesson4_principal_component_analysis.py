from lesson2_vectorize import get_words_matrix
import numpy as np
from sklearn.decomposition import PCA

def main():
    words_matrix = get_words_matrix()
    nnz = words_matrix.nnz
    total = words_matrix.shape[0] * words_matrix.shape[1]
    density = nnz / total
    sparsity = 1 - density
    sparse_memory_size = words_matrix.data.nbytes + words_matrix.indices.nbytes + words_matrix.indptr.nbytes
    print("Before todense() - SPARSE matrix:")
    print(f"Type: {type(words_matrix).__name__}")
    print(f"Shape: {words_matrix.shape}")
    print(f"Number of non-zero elements: {nnz}/{total}")
    print(f"Density: {density:.4f}")
    print(f"Sparsity: {sparsity:.4f}")
    print(f"Memory size (bytes): {sparse_memory_size}")
    print()

    words_matrix_dense = words_matrix.todense()
    nnz = np.count_nonzero(words_matrix_dense)
    total = words_matrix_dense.shape[0] * words_matrix_dense.shape[1]
    density = nnz / total
    sparsity = 1 - density
    print("After todense() - DENSE matrix:")
    print(f"Type: {type(words_matrix_dense).__name__}")
    print(f"Shape: {words_matrix_dense.shape}")
    print(f"Memory size (bytes): {words_matrix_dense.nbytes} > {sparse_memory_size}")
    print()
    print("-" * 100)

    # Преобразуем numpy.matrix в numpy.ndarray для совместимости с sklearn
    words_matrix_dense_array = np.asarray(words_matrix_dense)
    pca = PCA(n_components=2)
    pca.fit(words_matrix_dense_array)
    reduced_words_matrix_dense = pca.transform(words_matrix_dense_array)
    print(f"Reduced words matrix dense shape: {reduced_words_matrix_dense.shape}")
    print()

    print("Reduced words matrix dense:")
    print(reduced_words_matrix_dense)

if __name__ == "__main__":
    main()  