import pickle


with open('model_pickle', 'rb') as f:
    mp = pickle.load(f)
    
print(mp.predict([[3, 2, 4, 5]])    )