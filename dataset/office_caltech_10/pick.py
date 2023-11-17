import pickle


with open("dslr_test.pkl","rb") as fr:
    data = pickle.load(fr)
print(type(data[0]))
print(len(data[0]))
print(len(data[1]))