import matplotlib 
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from os import listdir

if __name__ == "__main__": 
    l = listdir('./matrices')
    matrices = []
    plt.ion()
    for elem in l:
        matrices.append(np.load('./matrices/'+elem))
    for elem in matrices:
        matrix_name = l.pop(0)
        print(matrix_name)
        shape = (elem.shape[0], elem.shape[1]+1)
        append_matrix = np.zeros((shape[0],1))
        new_matrix = np.concatenate((elem,append_matrix), axis = 1)
        for i in range(shape[0]):
            plt.imshow(elem[i,:].reshape(20,20))
            tag = input('1 for letter 0 for no letter')
            if tag == '1':
                new_matrix[i,-1]=1
            plt.close()
        np.save('./matrices/classified_'+ matrix_name, new_matrix)
    

        