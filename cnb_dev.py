# Python 3.6.8
# adam note: '##' used for testing in pycharm
##
# standard library
import math
import re
# third-party
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
# with naive_bayes_mod, predict(return_n=3) returns tuple(top1, [top3, top2, top1])
from sklearn.naive_bayes_mod import ComplementNB
# from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score
# first-party
import conndb

# --- set global vars ---
SAMPLE_PCT = 0.50
TRAIN_PCT = 0.70
# -----------------------

# --- connect to dev db ---
cur = conndb.connect_db()
# -------------------------


class MasterHS(object):
    """Initialize with MasterHS.construct()."""

    def __init__(self, counts):
        """Holds information about the 4-digit HS codes we are trying to classify.

        Generally, 'HS' will refer to 4-digit codes and 'HS2' will refer to the first 2 digits.

        Args:
            counts (np.array): nx2 with col1: str(hs code), col2: str(total # of obs)

        Attrs:
            counts (np.array): is above
            hs2counts (np.array): same idea as self.counts except col1: str(hs2 code)
            ptiles (np.array): nx1, contains percentiles of self.counts calculated using PERCENTILES
        """
        self.counts = counts
        self.hs2counts = None
        self.ptiles = None

    @classmethod
    def construct(cls):
        """Returns instance of the MasterHS class."""
        print("Creating MasterHS...")
        query = """
            select hs, count(*) as total
            from dev.zad_hs_results
            group by hs
            order by total
        """
        cur.execute(query)
        print("MasterHS created.")

        return cls(counts=np.array(cur.fetchall()))

    def define_hs2counts(self):
        hs2dict = {k[:2]: 0 for k in set(self.counts[:, 0])}
        print(hs2dict)
        
        for i in range(self.counts.shape[0]):
            hs2dict[self.counts[i, 0][:2]] += int(self.counts[i, 1])
        
        hs2counts = pd.DataFrame(hs2dict.items())
        print(hs2counts)
        
        hs2counts.columns = ['hs2', 'count']
        hs2counts['weight'] = hs2counts['count'] / hs2counts['count'].sum() * 100
        print(hs2counts.sort_values('weight', ascending=False))
        
        self.hs2counts = hs2counts.sort_values('weight', ascending=False)


class MasterDataFrame(object):
    """Initialize with MasterDataFrame.construct(MasterHS.counts)."""
    
    def __init__(self, dataframe):
        """Holds a stratified random sample of container description data (and subsets thereof).

        Args:
            dataframe (pd.DataFrame): nx7 array passed from internal construction query
            
        Attrs:
            original (pd.DataFrame): is above; original copy
            master (pd.DataFrame): copy of self.original
            train (pd.DataFrame): subset of self.master
            test (pd.DataFrame): subset of self.master, complement of self.train
            kfolds (dict): with keys: int(k value), values: list(index values)
            get_k (int): # of folds
        """
        self.original = dataframe
        self.master = dataframe
        self.train = None
        self.test = None
        self.kfolds = {}
        self.get_k = None

    @classmethod
    def construct(cls, hs_counts):
        """Returns instance of the MasterDataFrame class constructed from dev db query.

        Args:
            hs_counts: this should be MasterHS.counts
        """
        print("Creating MasterDataFrame...")
        query = """
            drop table if exists t_df;
                create temp table t_df(
                    description_id varchar, 
                    description varchar, 
                    port_origin varchar, 
                    port_us varchar,
                    shipper varchar,
                    consignee varchar,
                    hs varchar
                )
        """
        cur.execute(query)
        
        for i in range(hs_counts.shape[0]):
            hs_code = hs_counts[i, 0]
            hs_sample_obs = math.ceil(SAMPLE_PCT * int(hs_counts[i, 1]))
            query = """
                drop table if exists t_construct;
                    create temp table t_construct as (
                        select *
                        from dev.zad_construct_full
                        where hs = \'{}\'
                        limit {}
                    )
                ;
                insert into t_df
                select * from t_construct
                ;
                """.format(hs_code, hs_sample_obs)
            cur.execute(query)
        
        query = "select * from t_df"
        cur.execute(query)
        print("Data constructed from stratified sample.")
        
        colnames = ['desc_id', 'desc', 'port_origin', 'port_us', 'shipper', 'consignee', 'hs']
        return cls(dataframe=pd.DataFrame.from_records(cur.fetchall(), columns=colnames))

    def add_hs2(self):
        self.master['hs2'] = self.master['hs'].str[:2]
        print("New column generated for df.master: \'hs2\'")

    def add_desc_cat(self):
        # potential colnames = ['desc', 'port_origin', 'port_us', 'shipper', 'consignee']
        colnames = ['desc', 'consignee']
        self.master.fillna('', inplace=True)

        new_col = self.master[colnames[0]].values
        for name in colnames[1:]:
            new_col += ' ' + self.master[name].values

        self.master['desc_cat'] = new_col
        print("New column generated for df.master: \'desc_cat\'")

    @staticmethod
    def add_desc_nohs(hs_code, desc, sdw_only=False):
        """Strips the parsed HS code from description text. Potentially also strips [^A-za-z].

        Passed to lambda function as an argument for pd.DataFrame.apply() to generate 'desc_nohs'.

        Args:
            hs_code (str): 4-digit HS code
            desc (str): description text; element of self.master['desc_cat']
            sdw_only (bool): only keep whitespace (s), digits (d), word (w) characters

        Returns:
            stripped down version of desc (str)
        """
        pattern = r'[^\d]' + hs_code + r'([^\d]|\d{2}[^\d]|\d{4}[^\d]|\d{6}[^\d])'
        desc_nohs = re.sub(pattern, '', desc)
        if sdw_only:
            desc_nohs = re.sub(r'[^\s\d\w]', ' ', desc_nohs)
        return desc_nohs.lower()

    def add_metacode(self):
        """Adds 'metacode' col to df.master.

        Contains the metacode, which is predicted at level0 classification.
        Right now, metacodes are the first digit of the cargo's HS code.
        """
        self.master['metacode'] = self.master['hs'].str[:1]
        print("New column generated for df.master: \'metacode\'")

    def get_train_test(self, hs_counts):
        """Splits MasterDataFrame into subsets for training and testing based on global constants.

        Args:
            hs_counts (np.array): should be MasterHS.counts
        """
        print("Splitting df.master index into train/test subsets.")
        train_index = []
        for i in range(hs_counts.shape[0]):
            hs_code = hs_counts[i, 0]
            hs_count = int(hs_counts[i, 1])
            # hs_obs = math.ceil(hs_count * SAMPLE_PCT * TRAIN_PCT)  # use this when duplicates are not dropped
            hs_obs = math.ceil(hs_count * TRAIN_PCT)  # use this when dropping duplicates, as hs.counts is recalulated
            query = "hs == \'{}\'".format(hs_code)
            train_index += list(self.master.query(query).sample(n=hs_obs).index)
        train_index.sort()
        
        self.train = self.master.loc[train_index]
        self.test = self.master[~self.master.isin(self.train)].dropna(how='all')
        return print("Train/test split complete.")

    def get_strat_kfolds(self, folds=5, shuffle=False):
        """Alternative to self.get_train_test() that defines self.kfolds and self.get_k.
        TODO: still testing this, will not yet work with classification methods in MasterModel
        Access individual folds through:
            self.kfolds[k]			where k is an int in range(self.get_k)
            self.kfolds[k][0]		use 0 for X data and 1 for y data
            self.kfolds[k][0][0]	use 0 for train_index and 1 for test_index

        Args:
            folds (int): number of folds
            shuffle (bool): shuffles data before splitting into folds
        """
        skf = StratifiedKFold(n_splits=folds, shuffle=shuffle)
        self.get_k = skf.get_n_splits(self.master['desc_nohs'], self.master['hs'])

        k = 0
        for train_index, test_index in skf.split(self.master['desc_nohs'], self.master['hs']):
            self.kfolds[k] = [
                [self.master['desc_nohs'].loc[train_index], self.master['desc_nohs'].loc[test_index]],
                [self.master['hs'].loc[train_index], self.master['hs'].loc[test_index]]
            ]
            k += 1
        print("Stratified {}-folds generated.".format(folds))

    def add_c_hat_codes(self):
        """Adds codes to df.test that tell you if level0 yields a correct prediction or not."""
        right_subset = self.test.query("metacode == metapred")
        wrong_subset = self.test.query("metacode != metapred")
        rights = pd.Series(data=[1] * right_subset.shape[0], index=right_subset.index)
        wrongs = pd.Series(data=[0] * wrong_subset.shape[0], index=wrong_subset.index)
        c_hat_codes = pd.Series(rights).append(wrongs)
        
        self.test['c_hat_code'] = c_hat_codes


class MasterModel(object):
    def __init__(self):
        """Holds all of the classifiers, their results, and related methods.

        Attrs:
            models (dict): with keys: 'level0', int(all meta codes), str(all hs2 codes)
                                values: [classifier, tfidf vectorizer]
            results (dict): has the same keys as self.models
                                values: [predicted values, indices of tested data, accuracy]
            stop_words (dict): has the same keys as self.models
                                values: [list of stop words]
            hs2_branches (dict): with keys: str(all hs2 codes) and values: int(branch count at node).
                                in other words, the number of 4-digit hs codes that stem from each hs2 code.
        """
        self.models = {}
        self.results = {}
        self.stop_words = {}
        self.hs2_branches = {}

    def level0_classification(self, train_data, test_data):
        """Takes in text data from 'desc_nohs' and predicts meta codes for each row.

        The first round of classification is done with a single classifier.

        Args:
            train_data (MasterDataFrame.train)
            test_data (MasterDataFrame.test)
        """
        print("Start of level0.")
        x_train = train_data['desc_nohs']
        y_train = train_data['metacode']
        x_test = test_data['desc_nohs']
        y_test = test_data['metacode']

        if 'level0' in self.stop_words:
            stopwords = self.stop_words['level0']
        else:
            stopwords = None

        tfidf_vec = TfidfVectorizer(
            strip_accents='ascii',
            lowercase=True,
            preprocessor=None,
            tokenizer=None,
            analyzer='word',
            stop_words=stopwords,
            smooth_idf=True,
            sublinear_tf=True,
            ngram_range=(2, 2),
            max_df=0.25,
            min_df=1
        )
        x_train_tfidf = tfidf_vec.fit_transform(x_train, y_train)
        x_test_tfidf = tfidf_vec.transform(x_test)

        clf = ComplementNB(alpha=0.05, norm=False)
        clf.fit(x_train_tfidf, y_train)
        metacode_preds = clf.predict(x_test_tfidf)
        
        acc = accuracy_score(y_test, metacode_preds)
        print("level0 classification accuracy: {}".format(acc))

        test_data['metapred'] = metacode_preds
        print("New column generated for df.test: \'metapred\'")

        self.models['level0'] = [clf, tfidf_vec]
        self.results['level0'] = [metacode_preds, 'No need for index', acc]

    def level1_classification(self, level1_codes, train_data, test_data):
        """Takes in text data from 'desc_nohs' and predicts 2-digit HS codes.

        The second round of classification is done by using different classifiers for each subset of data
            partitioned by meta code.

        Args:
            level1_codes (sequence): a list of whatever we want to predict with this method
            train_data (MasterDataFrame.train)
            test_data (MasterDataFrame.test)
        """
        print("Start of level1.")
        for level1_code in set(level1_codes):
            query = "metacode == \'{}\'".format(level1_code)
            train_subset = train_data.query(query)
            test_subset = test_data.query(query)

            if level1_code in self.stop_words:
                stopwords = self.stop_words[level1_code]
            else:
                stopwords = None

            tfidf_vec = TfidfVectorizer(
                strip_accents='ascii',
                lowercase=True,
                preprocessor=None,
                tokenizer=None,
                analyzer='word',
                stop_words=stopwords,
                smooth_idf=True,
                sublinear_tf=True,
                ngram_range=(2, 2),
                max_df=1.00,
                min_df=1
            )
            x_train_tfidf = tfidf_vec.fit_transform(train_subset['desc_nohs'], train_subset['hs2'])
            x_test_tfidf = tfidf_vec.transform(test_subset['desc_nohs'])

            clf = ComplementNB(alpha=0.05, norm=False)
            clf.fit(x_train_tfidf, train_subset['hs2'])
            hs2_preds = clf.predict(x_test_tfidf)

            acc = accuracy_score(test_subset['hs2'], hs2_preds)
            print("level1 {} classifier accuracy: {}".format(level1_code, acc))

            self.models[level1_code] = [clf, tfidf_vec]
            self.results[level1_code] = [hs2_preds, test_subset.index, acc]

    def level2_classification(self, hs2_codes, train_data, test_data):
        """Takes in text data from 'desc_nohs' and predicts 4-digit HS codes.
        The final round of classification is done by using different classifiers for each subset of data
            partitioned by 2-digit HS code.
        """
        print("Start of level2.")
        for hs2_code in set(hs2_codes):
            print('log: ', hs2_code)
            query = "hs2 == \'{}\'".format(hs2_code)
            train_subset = train_data.query(query)
            test_subset = test_data.query(query)

            if hs2_code in self.stop_words:
                stopwords = self.stop_words[hs2_code]
            else:
                stopwords = None

            tfidf_vec = TfidfVectorizer(
                strip_accents='ascii',
                lowercase=True,
                preprocessor=None,
                tokenizer=None,
                analyzer='word',
                stop_words=stopwords,
                smooth_idf=True,
                sublinear_tf=True,
                ngram_range=(2, 2),
                max_df=1.00,
                min_df=1
            )
            x_train_tfidf = tfidf_vec.fit_transform(train_subset['desc_nohs'], train_subset['hs'])
            x_test_tfidf = tfidf_vec.transform(test_subset['desc_nohs'])

            clf = ComplementNB(alpha=0.05, norm=False)
            clf.fit(x_train_tfidf, train_subset['hs'])

            n = self.hs2_branches[hs2_code]
            n = 3 if n > 3 else n
            hs4_preds = clf.predict(x_test_tfidf, return_n=n)
            # TODO: error with argpartition, check tomorrow
            acc = accuracy_score(test_subset['hs'], pd.Series(hs4_preds[0]))
            print("HS2 Code {} classifier accuracy: {}".format(hs2_code, acc))

            self.models[hs2_code] = [clf, tfidf_vec]
            self.results[hs2_code] = [hs4_preds[0], test_subset.index, acc, hs4_preds[1]]

    def multihs_classification(self, original_data):
        print("Start of multiple hs classification.")
        nodups = original_data.drop_duplicates(subset='desc_id', keep=False)
        dups = original_data[~original_data.isin(nodups)].dropna(how='all')
        print("Constructing 'multihs' binary variable...")
        dups = pd.Series([1] * len(dups), index=dups.index)
        nodups = pd.Series([0] * len(nodups), index=nodups.index)
        print("Appending...")
        multihs_binary = dups.append(nodups).sort_index()
        original_data['multihs'] = multihs_binary
        print("New column generated for df.original: 'multihs'")

        train_data = original_data.sample(frac=TRAIN_PCT)
        test_data = original_data[~original_data.isin(train_data)].dropna(how='all')
        x_train = train_data['desc_nohs']
        y_train = train_data['multihs']
        x_test = test_data['desc_nohs']
        y_test = test_data['multihs']

        tfidf_vec = TfidfVectorizer(
            strip_accents='ascii',
            lowercase=True,
            preprocessor=None,
            analyzer='word',
            stop_words=None,
            smooth_idf=True,
            sublinear_tf=True,
            ngram_range=(2, 2),
            max_df=0.50,
            min_df=1
        )
        x_train_tfidf = tfidf_vec.fit_transform(x_train, y_train)
        x_test_tfidf = tfidf_vec.transform(x_test)

        clf = ComplementNB(alpha=0.01, norm=False)
        clf.fit(x_train_tfidf, y_train)
        multihs_preds = clf.predict(x_test_tfidf)
        print(multihs_preds)

        acc = accuracy_score(y_test, multihs_preds)
        print("Multiple HS classification accuracy: {}".format(acc))

        self.models['multihs'] = [clf, tfidf_vec]
        self.results['multihs'] = [multihs_preds, 'No need for index', acc]

    @staticmethod
    def gen_prediction_col(some_code_array, results_dict, rd_idx):
        """Aligns the index of predictions from many different classifiers.

        Args:
            some_code_array: intended to be either MasterDataFrame.test['hs2pred'] or MasterDataFrame.test['hs4pred']
            results_dict: intended to be MasterModel.results
            rd_idx: which data to access from results_dict
                        0: normal predicted values
                        3: lists of n-most-likely predicted values

        Returns:
            pd.Series object with index matching MasterDataFrame.test
        """
        print("Generating prediction column...")
        prediction_col = pd.Series()
        for code in set(some_code_array):
            pred_subset = pd.Series((x for x in results_dict[code][rd_idx]), index=results_dict[code][1])
            prediction_col = prediction_col.append(pred_subset)

        print("Returning prediction column:\n", prediction_col)
        return prediction_col

    def live_test(self, level0_preds, test_data):
        for level0_code in set(level0_preds):
            subset = test_data.query("metapred == \'{}\'".format(level0_code))
            x_test_tfidf = self.models[level0_code][1].transform(subset['desc_nohs'])
            hs2_preds = self.models[level0_code][0].predict(x_test_tfidf)

            acc = accuracy_score(subset['hs2'], hs2_preds)
            print("level1 {} classifier accuracy: {}".format(level0_code, acc))

            self.results[level0_code] = [hs2_preds, subset.index, acc]

        test_data['hs2pred'] = MasterModel.gen_prediction_col(level0_preds, self.results, 0)
        print("New column generated for df.test: \'hs2pred\'")

        for hs2_code in set(test_data['hs2pred']):
            subset = test_data.query("hs2pred == \'{}\'".format(hs2_code))
            x_test_tfidf = self.models[hs2_code][1].transform(subset['desc_nohs'])
            n = self.hs2_branches[hs2_code]
            n = 3 if n > 3 else n
            hs4_preds = self.models[hs2_code][0].predict(x_test_tfidf, return_n=n)

            acc = accuracy_score(subset['hs'], hs4_preds[0])
            print("HS2 Code {} classifier accuracy: {}".format(hs2_code, acc))

            self.results[hs2_code] = [hs4_preds[0], subset.index, acc, hs4_preds[1]]
        print('log hs4pred:\n', self.results['84'][0])
        print('log hs4pred:\n', len(self.results['84'][0]))
        test_data['hs4pred'] = MasterModel.gen_prediction_col(test_data['hs2pred'], self.results, 0)
        print('log hs4pred:\n', self.results['84'][3])
        print('log hs4pred:\n', len(self.results['84'][0]))
        test_data['hs4pred_list'] = MasterModel.gen_prediction_col(test_data['hs2pred'], self.results, 3)
        live_acc = accuracy_score(test_data['hs'], test_data['hs4pred'])

        self.results['live'] = live_acc
        print("Live test accuracy: ", self.results['live'])

    def avg_hs2_acc(self, test_data):
        total = 0
        for hs2_code in set(test_data['hs2pred']):
            total += self.results[hs2_code][2]
        print(total / len(set(test_data['hs2pred'])))

    def persist(self, *, name):
        file_name = "/home/adam/text_class/models/" + name + ".joblib"
        print("Compressing...")
        dump(self.models, file_name, compress=3, protocol=4)
        print("MasterModel.models dictionary now persists at:\n\t{}".format(file_name))
        
    def live_prediction(self, live_data):
        print("Start of multihs classification.")
        x_tfidf_multihis = self.models['multihs'][1].transform(live_data['desc_cat'])
        multihis_preds = self.models['multihis'][0].predict(x_tfidf_multihis)

        live_data['multihs_pred'] = multihis_preds
        print("New column generated for df: \'multihs_pred\'")

        print("Start of level0.")
        x_tfidf_level0 = self.models['level0'][1].transform(live_data['desc_cat'])
        meta_preds = self.models['level0'][0].predict(x_tfidf_level0)

        live_data['metapred'] = meta_preds
        print("New column generated for df: \'metapred\'")

        for level0_code in set(live_data['metapred']):
            subset = live_data.query("metapred == \'{}\'".format(level0_code))
            x_tfidf_level1 = self.models[level0_code][1].transform(subset['desc_cat'])
            hs2_preds = self.models[level0_code][0].predict(x_tfidf_level1)

            self.results[level0_code] = [hs2_preds, subset.index]

        live_data['hs2pred'] = MasterModel.gen_prediction_col(live_data['metapred'], self.results)
        print("New column generated for df: \'hs2pred\'")

        for hs2_code in set(live_data['hs2pred']):
            subset = live_data.query("hs2pred == \'{}\'".format(hs2_code))
            x_test_tfidf = self.models[hs2_code][1].transform(subset['desc_cat'])
            n = self.hs2_branches[hs2_code]
            n = 3 if n > 3 else n
            hs4_preds = self.models[hs2_code][0].predict(x_test_tfidf, return_n=n)

            self.results[hs2_code] = [hs4_preds[0], subset.index]

        live_data['hs4pred'] = MasterModel.gen_prediction_col(live_data['hs2pred'], self.results, 0)
        live_data['hs4pred_list'] = MasterModel.gen_prediction_col(live_data['hs2pred'], self.results, 3)
        print("New column generated for df: \'hs4pred\'")

        return x_tfidf_level0

    def calc_c_hat(self, test_data):
        """Calculates c_hat by taking the dot product of tf-idf and feature_log_prob.

        Args:
            test_data: currently only supports df.test (level0)

        Returns:
            adds float(c_hat) col to test_data
        """
        clf_meta = self.models['level0'][0]
        x_tfidf_meta = self.models['level0'][1].transform(test_data['desc_nohs'])
        flp = clf_meta.feature_log_prob_
        meta_code_to_flp_key = {v: i for i, v in enumerate(clf_meta.classes_)}

        c_hats = []
        for i in range(test_data.shape[0]):
            meta_code = test_data['metapred'].iloc[i]
            flp_key = meta_code_to_flp_key[meta_code]
            c_hat = float(x_tfidf_meta[i, :].dot(flp[flp_key, :]))
            c_hats.append(c_hat)

        return c_hats

    def read_stop_words(self):
        # TODO: test this, i'll probably have to change the orientation of the data
        stopwords = pd.read_csv('/home/adam/text_class/data/STOP_WORDS.csv', columns=['code', 'stop_words'])
        self.stop_words = stopwords.to_dict(orient='records')

    def define_hs2branches(self, train_data):
        """Just takes in the training data, maybe this can be sped up later"""
        df = train_data
        branch_counts = {}
        for hs2_code in set(df['hs2']):
            branch_counts[hs2_code] = len(set(df.query("hs2 == '{}'".format(hs2_code))['hs']))
        self.hs2_branches = branch_counts
        print("Branch counts defined for each HS2 node: hs.hs2branches")


def data_management():
    hs = MasterHS.construct()
    df = MasterDataFrame.construct(hs.counts)
    df.add_hs2()
    df.add_desc_cat()
    # TODO: low priority: slowest method in data_management, not sure if we can speed up anymore
    df.master['desc_nohs'] = df.master.apply(
        lambda row: df.add_desc_nohs(row['hs'], row['desc_cat'], sdw_only=True),
        axis=1
    )
    print("New column generated for df.master: 'desc_nohs'")
    df.add_metacode()
    print("Starting size: ", df.master.shape)
    df.master.drop_duplicates(subset='desc_id', keep=False, inplace=True)
    print("Dropped duplicates")
    print("New size: ", df.master.shape)
    hs.counts = np.column_stack(
        (df.master['hs'].value_counts().index.values,
         df.master['hs'].value_counts().values)
    )
    print("hs.counts recalculated as actual sample counts.")
    df.get_train_test(hs.counts)

    return df, hs


def model_development(master_classes):
    df, hs = master_classes
    model = MasterModel()
    model.define_hs2branches(df.train)
    model.multihs_classification(df.original)
    model.level0_classification(
        train_data=df.train,
        test_data=df.test
    )
    model.level1_classification(
        train_data=df.train,
        test_data=df.test,
        level1_codes=df.train['metacode']
    )
    model.level2_classification(
        train_data=df.train,
        test_data=df.test,
        hs2_codes=df.train['hs2'],
    )
    print("Start of live test.")
    model.live_test(
        level0_preds=df.test['metapred'],
        test_data=df.test,
    )
    model.avg_hs2_acc(
        test_data=df.test
    )
    # model persistence
    # model.persist(name='cnb_v2')
    return df, hs, model


def in_the_works():
    """Probably lots of red in this atm..."""
    # calculating c_hat without transforming sparse matrix
    df.test['c_hat'] = model.calc_c_hat(df.test)
    df.add_c_hat_codes()

    # plotting c_hat distribution
    c_hat_results = df.test[['metacode', 'metapred', 'c_hat', 'c_hat_code']]
    c_hat_results.to_csv("/home/adam/text_class/data/c_hat_results1.csv")

    # TODO: work through this and implement as methods in MasterModel / MasterDataFrame
    # CONFIDENCE ADJUSTMENT CODE
    # TODO: fix code below
    #  percentile_cutoff = np.percentile(c_hat_results_pd['right'], 80)


    def perc_adjust(row):
        c_hat_key = row['c_hat']
        if c_hat_key > percentile_cutoff:
            return row['desc_id']


    cut_desc_ids = list(df.test.apply(lambda row: perc_adjust(row)))
    # adam notes
    # sparse.csr_matrix.data seems like it gives you all of the non_zero values in the matrix.
    # sparse.csr_matrix.count_nonzero() does what it says.
    # removing stopwords by looking at the tf-idf distribution
    # looking at the training tfidf to see what words we can cut
    print("Calculating the median inverse document frequency...")
    tfidf = pd.Series(tfidf_vec_meta.idf_)
    tfidf = tfidf[tfidf < np.percentile(tfidf, 75)]
    print("Filtering bigrams...")
    stop_words_index = list(tfidf.index)
    tfidf_vocab_dict = tfidf_vec_meta.vocabulary_
    tfidf_vocab_dict = {v: k for k, v in tfidf_vocab_dict.items()}

    stop_words = []
    for i in stop_words_index:
        stop_word = tfidf_vocab_dict[i]
        stop_words.append(stop_word)
    print("Stop words available as list object: 'stop_words'")


##
def main():
    return model_development(data_management())


if __name__ == '__main__':
    DF, HS, MODEL = main()


##
def debug_container():
    """Return the 10 most influential bigrams at each level of classification."""
    clf_debug = MODEL.models['8'][0]
    vec_debug = MODEL.models['8'][1]
    # flp has probs for each feature for each class
    flp = clf_debug.feature_log_prob_
    flp = pd.DataFrame(flp)
    # inverting the vocab dict such that we have
    #   key(integer): value(bigram)
    vocab = {value: key for key, value in vec_debug.vocabulary_.items()}
    # using our new vocab to map bigrams to feature colnames in flp
    colnames = flp.columns
    colnames = [vocab[x] for x in colnames]
    flp.columns = colnames
    # now flp is of the form a x b
    # where a is the classes and b is the features
    # now we must take a look at the tfidf
    print("Transforming...")
    tfidf = vec_debug.transform(DF.test.query("metacode == '8'")['desc_nohs'])
    # tfidf is a sparse matrix of the form
    #   n_samples x n_features
    # so now for the final piece, take the dot product
    # of tfidf and flp.T (transposed)
    jll = safe_sparse_dot(tfidf, flp.to_numpy().T)
    # DEBUG BREAKDOWN
    # first we need to identify which container we want to debug
    # by its row index number in DF
    tfidf_debug = tfidf[2, :]
    # change to numpy from sparse matrix
    tfidf_debug = tfidf_debug.toarray()
    # find c_hats for all classes
    jll_debug = flp.to_numpy() * tfidf_debug
    # make it a dataframe, map colnames, then remove all 0 columns
    jll_debug = pd.DataFrame(jll_debug)
    colnames = jll_debug.columns
    colnames = [vocab[x] for x in colnames]
    jll_debug.columns = colnames
    print("Dropping 0s...")
    # praise stack overflow for the super fast code below that drops 'all 0' features
    jll_debug = jll_debug.loc[:, (~jll_debug.isin([0, 1])).any(axis=0)]
    # now we need to narrow this down to the <= 10 most influential features
    # for the class that was predicted
    # for now, I will add a TOTAL column at the end and sort by it
    jll_debug['$C_i$'] = jll_debug.sum(axis=1)
    jll_debug = jll_debug.sort_values('$C_i$', ascending=False)

    return jll_debug


JLL = debug_container()

## 
# from sklearn.utils
def safe_sparse_dot(a, b, dense_output=False):
    """Dot product that handle the sparse matrix case correctly

    Uses BLAS GEMM as replacement for numpy.dot where possible
    to avoid unnecessary copies.

    Parameters
    ----------
    a : array or sparse matrix
    b : array or sparse matrix
    dense_output : boolean, default False
        When False, either ``a`` or ``b`` being sparse will yield sparse
        output. When True, output will always be an array.

    Returns
    -------
    dot_product : array or sparse matrix
        sparse if ``a`` or ``b`` is sparse and ``dense_output=False``.
    """
    from scipy import sparse
    if sparse.issparse(a) or sparse.issparse(b):
        ret = a * b
        if dense_output and hasattr(ret, "toarray"):
            ret = ret.toarray()
        return ret
    else:
        return np.dot(a, b)
