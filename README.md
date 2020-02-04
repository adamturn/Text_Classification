# Text Classification
## Python scripts written to run on a CentOS server with Python 3.5.4. Yum!
### cnb_dev.py is a sandbox to train and test tree-based text classification models that predict Harmonized Scheduling codes by analyzing containerized cargo description data. After basic natural language processing, data is passed to the head node where it is vectorized and used to train the scikit-learn*1 implementation of a Complement Naive Bayes*2 classifier, which is also placed at each sequential node.
#### *https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html
#### *Rennie et al. (2003) https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf
### predict_hs.py provides a user-friendly interface for the application of an exported model from cnb_dev.py to live data through a SQL query.
