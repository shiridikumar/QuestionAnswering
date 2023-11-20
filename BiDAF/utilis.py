def load_glove_dict(file_path):
    glove_dict = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            glove_dict[word] = vector
    return glove_dict

def create_weights_matrix(glove_dict, word_vocab):
    weights_matrix = np.zeros((len(word_vocab), 100))
    words_found = 0
    for i, word in enumerate(word_vocab):
        try:
            weights_matrix[i] = glove_dict[word]
            words_found += 1
        except KeyError:
            pass
    return weights_matrix, words_found

##usage
# glove_dict = load_glove_dict("./data/glove.6B.100d.txt")
# weights_matrix, words_found = create_weights_matrix(glove_dict, word_vocab)
# print("Words found in the GloVe vocab:", words_found)
# np.save('bidafglove.npy', weights_matrix)