import numpy as np

word_embedding = [
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12]
]

def give_pos_vec(word_vec):
    pos = len(word_vec)
    index = len(word_vec[0])
    pos_vec = []

    for i in range(pos):
        vec = []

        for j in range(index):
            if j % 2 == 0:
                vec.append(np.sin(i / 10000**(j/index)))
            else:
                vec.append(np.cos(i / 10000**((j-1)/index)))

        pos_vec.append(vec)

    return np.matrix(pos_vec)

print(give_pos_vec(word_embedding))