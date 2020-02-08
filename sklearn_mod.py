from shutil import copyfile
import fileinput


def naive_bayes_mod(src_path):
    """Creates a modified copy of naive_bayes.py in sklearn package.

    Adds functionality to all classes that inherit from _BaseNB.
    Setting clf.predict(return_n=int) causes clf.predict() to return a tuple:
        np.array(predictions), np.array(list[n_highest_predictions])
    n_highest_predictions is sorted in ascending order: eh-->maybe-->probably
    Remember to reference the modified version in import statements:
        >>> from sklearn.naive_bayes_mod import ComplementNB()

    Args:
        src_path (str): path to sklearn's naive_bayes.py for current project
            note: this is usually '.../site-packages/sklearn/naive_bayes.py'
    """
    mod_path = src_path.replace('naive_bayes', 'naive_bayes_mod')
    copyfile(src_path, mod_path)

    mod_map = {
        'def predict(self, X):':
            '\tdef predict(self, X, return_n=False):\n',
        'return self.classes_[np.argmax(jll, axis=1)]':
            '\t\tif return_n:\n\t\t\treturn self.classes_[np.argmax(jll, axis=1)], ' +
            'self.classes_[np.argpartition(jll, -return_n, axis=1)[:, -return_n:]]\n' +
            '\t\telse:\n\t\t\treturn self.classes_[np.argmax(jll, axis=1)]\n'
    }

    for line in fileinput.input(mod_path, inplace=True):
        if line.strip() in mod_map:
            print('{}'.format(mod_map[line.strip()]), end='')
        else:
            print(line, end='')

    return print("sklearn.naive_bayes_mod is now available.")
