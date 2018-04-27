import pickle as pkl

from preprocess import make_vectorizer, preprocess

def main():
    main_list = []
    raw_text = pkl.load(open('data/corpus.p', 'rb'))
    for i in range(len(raw_text)):
        tmp_list = preprocess(raw_text[i])
        main_list.append(tmp_list)
    make_vectorizer(main_list)

if __name__ == '__main__':
    main()
