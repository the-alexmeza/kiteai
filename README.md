# Kite AI
Development on Kite AI. Code is **NOT** the version before we dissolved,
rather a personal derivative to continue onwards.

This is based on the Kaggle competition [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).
It uses their dataset for initial training, in the future I will phase in data from Kite AI.

---

### What is this?

Kite AI is an idea to create a universal tool to protect against online harassment
and cyberbullying. Initially, it began as a hackathon project at [HackGSU 2017](http://hackgsu.com)
where it won the Hack Harassment reward. From then, Justin Potts([@justinpotts](https://github.com/justinpotts)), Alex Meza ([@the-alexmeza](https://github.com/the-alexmeza)),
and Trevor "Thai" Nguyen ([@trevortknguyen](https://github.com/trevortknguyen)) continued development, and became a startup company.

After one year, it was decided that the optimal route to continue would be to open source the project
to allow for other contribution. Cyberbullying and online harassment are global problems, so it
only makes sense to allow anyone and everyone to contribute to the development.

This repository specifically is not the version of Kite AI that us three ended on, but a
personal derivation updating our method based on the experience I have gained in the meanwhile.
There are some aspects of the original code, namely the preprocessing and split_data.py, however anything else
is my experimentation.

This repository is **not** API-ready. `train_model.py` is not set up, and is based off a separate Keras project.

---

### File Explanation

- `make_dict.py` : Generates vocabulary (7000 words), makes files `data/vocab.p` and `data/corpus.p`.
- `make_vectorizer.py` : Generates a scikit-learn CountVectorizer (Likely changing). Makes file `data/vectorizer.p`
- `preprocess.py` : Holds text preprocessing and conversion to sparse matrices.
- `split_data.py` : Generates files `data/train_labels.p`, `data/train_samples.p`, `data/test_labels.p`, `data/test_samples.p`.
- `train_model.py` : Trains and saves model.
- `train.csv` : Kaggle provided dataset.

---

### Contributing

Just create a branch with a descriptive title. Completed models can be pushed into `models/your_model/`.

Please include accuracy scores (Precision, Recall, F1), and name your model folder after the model architecture (i.e. `models/NB_SVM_ensemble/`).

I will contact you to ask whether or not you want your model to be included in the master branch.

Any questions or comments can be sent to be at alex@mezainnovations.com.

---
### Todo

- Provide requirements.txt
- Complete prototype `train_model.py`.
- Upload `test.csv` upon model generation.
- Add to Issues tab.
- Add PR/Issues template.

---

### Other questions?

Whether you have questions, comments, or concerns regarding this project, or if you simply want to chat,
feel free to contact me at alex@mezainnovations.com.
