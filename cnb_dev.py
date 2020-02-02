# adam note: '##' used for testing in pycharm
##
# built-in
import math
import re
# external
import psycopg2
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score

# --- set constants ---
SAMPLE_PCT = 1.00  # percent of total obs sampled for each HS code
TRAIN_PCT = 0.70  # percent of sampled obs used for training (complement used for testing)
PERCENTILES = (99.5,)  # percentiles that determine volume codes
HOSTNAME = "xxx-xx-xxx-xxx-xx.compute-1.amazonaws.com"
# ---------------------

# --- connect to dev db ---
print("Establishing connection with database...")
conn = psycopg2.connect(
    host=HOSTNAME,
    database="db",
    port=5432,
    user="usr",
    password="pw"
)
cur = conn.cursor()
print("Connection established.")
# -------------------------


class MasterHS:
    """
    Initialize with MasterHS.construct().
    """
    def __init__(self, counts):
        """
        Holds information about the 4-digit HS codes we are trying to classify.
        Generally, 'HS' will refer to 4-digit codes and 'HS2' will refer to the first 2 digits.

        Args:
            counts (np.array): nx2 with col1: str(hs code), col2: str(total # of obs)

        Attrs:
            counts (np.array): is above.
            hs2counts(np.array): same idea as self.counts except col1: str(hs2 code)
            ptiles (np.array): nx1, contains percentiles of self.counts calculated using PERCENTILES
            vol (dict): with keys: str(hs code), values: int(volume code)
        """
        self.counts = counts
        self.hs2counts = None
        self.ptiles = None
        self.vol = None

    @classmethod
    def construct(cls):
        """
        Returns instance of the MasterHS class.
        """
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

    def define_vol(self, percentiles=PERCENTILES):
        """
        Defines volume codes (self.vol) based on percentiles of self.counts (self.ptiles).

        Args:
            percentiles (list): global constant PERCENTILES is passed automatically
        """
        print("Defining volume codes...")
        self.ptiles = np.percentile(self.counts[:, 1].astype(int), percentiles)
        self.vol = {}
        for i in range(self.counts.shape[0]):
            hs_code = self.counts[i, 0]
            count = int(self.counts[i, 1])
            if count > self.ptiles[-1]:
                self.vol[hs_code] = len(self.ptiles)
                continue
            else:
                for vol_code, ptile in enumerate(self.ptiles):
                    if count <= ptile:
                        self.vol[hs_code] = vol_code
                        break

        return print("Volume codes defined based on supplied percentiles.")

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


class MasterDataFrame:
    """
    Initialize with MasterDataFrame.construct(MasterHS.counts).
    """
    def __init__(self, dataframe):
        """
        Holds a stratified random sample of container description data (and subsets thereof).

        Args:
            dataframe (pd.DataFrame): nx7 array passed from internal construction query
        Attrs:
            master (pd.DataFrame): is above; original copy
            train (pd.DataFrame): subset of self.master
            test (pd.DataFrame): subset of self.master, complement of self.train
            kfolds (dict): with keys: int(k value), values: list(index values)
            get_k (int): # of folds
        """
        self.master = dataframe
        self.train = None
        self.test = None
        self.kfolds = {}
        self.get_k = None

    @classmethod
    def construct(cls, hs_counts):
        """
        Returns instance of the MasterDataFrame class constructed from dev db query.

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
        colnames = ['desc_id', 'desc', 'port_origin', 'port_us', 'shipper', 'consignee', 'hs']
        print("Data constructed from stratified sample and added to df.master")

        return cls(dataframe=pd.DataFrame.from_records(cur.fetchall(), columns=colnames))

    def add_hs2(self):
        self.master['hs2'] = self.master['hs'].str[:2]
        print("New column generated for df.master: \'hs2\'")

    def add_desc_cat(self):
        # colnames = ['desc', 'port_origin', 'port_us', 'shipper', 'consignee']
        colnames = ['desc', 'consignee']
        self.master.fillna('', inplace=True)
        new_col = self.master[colnames[0]].values
        for name in colnames[1:]:
            new_col += ' ' + self.master[name].values

        self.master['desc_cat'] = new_col
        print("New column generated for df.master: \'desc_cat\'")

    @staticmethod
    def add_desc_nohs(hs_code, desc, alpha_only=False):
        """
        Strips the parsed HS code from description text. Potentially also strips [^A-za-z].
        Passed to lambda function as an argument for pd.DataFrame.apply() to generate 'desc_nohs'.

        Args:
            hs_code (str): 4-digit HS code
            desc (str): description text; element of self.master['desc_cat']
            alpha_only (bool): whether or not to strip everything except alphabetic chars

        Returns:
            stripped down version of desc (str)
        """
        pattern = r'[^\d]' + hs_code + r'([^\d]|\d{2}[^\d]|\d{4}[^\d]|\d{6}[^\d])'
        desc_nohs = re.sub(pattern, ' ', desc)
        if alpha_only:
            desc_nohs = re.sub(r'[^A-Za-z]', ' ', desc_nohs)

        return desc_nohs.lower()

    def add_vol(self, vol_dict):
        """
        Adds volume code column to self.master.

        Args:
            vol_dict: should be MasterHS.vol
        """
        self.master['vol'] = self.master['hs'].map(vol_dict)
        print("New column generated for df.master: \'vol\'")

    def add_metacode(self):
        metacode_dict = {}
        for hs2_code in set(self.master['hs2']):
            base = 0
            for i in range(10):
                if int(hs2_code) in list(range(base, base + 10)):
                    metacode_dict[hs2_code] = base
                else:
                    base += 10

        self.master['metacode'] = self.master['hs2'].map(metacode_dict)
        print("New column generated for df.master: \'metacode\'")

    def get_train_test(self, hs_counts):
        """
        Splits MasterDataFrame into subsets for training and testing based on global constants.

        Args:
            hs_counts (np.array): should be MasterHS.counts
        """
        print("Splitting df.master index into train/test subsets.")
        train_index = []
        for i in range(hs_counts.shape[0]):
            hs_code = hs_counts[i, 0]
            hs_count = int(hs_counts[i, 1])
            # hs_obs = math.ceil(hs_count * SAMPLE_PCT * TRAIN_PCT)
            hs_obs = math.ceil(hs_count * TRAIN_PCT)  # use this when dropping duplicates, as hs.counts is recalulated
            query = "hs == \'{}\'".format(hs_code)
            train_index += list(self.master.query(query).sample(n=hs_obs).index)
        train_index.sort()
        self.train = self.master.loc[train_index]
        self.test = self.master[~self.master.isin(self.train)].dropna(how='all')

        return print("Train/test split complete.")

    def get_strat_kfolds(self, folds=5, shuffle=False):
        """
        TODO: still testing this, will not yet work with classification methods in MasterModel
        Alternative to self.get_train_test() that defines self.kfolds and self.get_k.
        Access individual folds through:
            self.kfolds[k]            where k is an int in range(self.get_k)
            self.kfolds[k][0]        use 0 for X data and 1 for y data
            self.kfolds[k][0][0]    use 0 for train_index and 1 for test_index

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


class MasterModel:
    def __init__(self):
        """
        Attrs:
            models (dict): with keys: 'level0', int(all volume codes), str(all hs2 codes)
                                values: [classifier, tfidf vectorizer]
            results (dict): has the same keys as self.models
                                values: [array of predicted values, indices of tested data, accuracy]
        """
        self.models = {}
        self.results = {}

    def level0_classification(self, train_data, test_data):
        """
        The first round of classification is done through a single classifier.
        Takes in text data from 'desc_nohs' and predicts volume codes for each row.
        """
        print("Start of level0.")
        x_train = train_data['desc_nohs']
        y_train = train_data['metacode']
        x_test = test_data['desc_nohs']
        y_test = test_data['metacode']
        tfidf_vec = TfidfVectorizer(
            strip_accents='ascii',
            lowercase=True,
            preprocessor=None,
            tokenizer=None,
            analyzer='word',
            stop_words=None,
            smooth_idf=True,
            sublinear_tf=True,
            ngram_range=(2, 2),
            max_df=0.75,
            min_df=1
        )
        x_train_tfidf = tfidf_vec.fit_transform(x_train, y_train)
        x_test_tfidf = tfidf_vec.transform(x_test)

        clf = ComplementNB(alpha=1, norm=True)
        clf.fit(x_train_tfidf, y_train)
        metacode_preds = clf.predict(x_test_tfidf)
        acc = accuracy_score(y_test, metacode_preds)
        print("level0 classification accuracy: {}".format(acc))

        test_data['metapred'] = metacode_preds
        print("New column generated for df.test: \'metapred\'")

        self.models['level0'] = [clf, tfidf_vec]
        self.results['level0'] = [metacode_preds, acc]

    def level1_classification(self, level1_codes, train_data, test_data):
        """
        The second round of classification is done by using different classifiers for each subset of data
            partitioned by volume code.
        Takes in text data from 'desc_nohs' and predicts 2-digit HS codes.
        """
        print("Start of level1.")
        for level1_code in set(level1_codes):
            query = "metacode == \'{}\'".format(level1_code)
            subset_train = train_data.query(query)
            subset_test = test_data.query(query)

            tfidf_vec = TfidfVectorizer(
                strip_accents='ascii',
                lowercase=True,
                preprocessor=None,
                tokenizer=None,
                analyzer='word',
                stop_words=None,
                smooth_idf=True,
                sublinear_tf=True,
                ngram_range=(2, 2),
                max_df=0.75,
                min_df=1
            )
            x_train_tfidf = tfidf_vec.fit_transform(subset_train['desc_nohs'], subset_train['hs2'])
            x_test_tfidf = tfidf_vec.transform(subset_test['desc_nohs'])

            clf = ComplementNB(alpha=1, norm=True)
            clf.fit(x_train_tfidf, subset_train['hs2'])
            hs2_preds = clf.predict(x_test_tfidf)
            acc = accuracy_score(subset_test['hs2'], hs2_preds)
            print("level1 {} classifier accuracy: {}".format(level1_code, acc))

            self.models[level1_code] = [clf, tfidf_vec]
            self.results[level1_code] = [hs2_preds, subset_test.index, acc]

    def level2_classification(self, hs2_codes, train_data, test_data):
        """
        The final round of classification is done by using different classifiers for each subset of data
            partitioned by 2-digit HS code.
        Takes in text data from 'desc_nohs' and predicts 4-digit HS codes.
        """
        print("Start of level2.")
        for hs2_code in set(hs2_codes):
            query = "hs2 == \'{}\'".format(hs2_code)
            subset_train = train_data.query(query)
            subset_test = test_data.query(query)

            tfidf_vec = TfidfVectorizer(
                strip_accents='ascii',
                lowercase=True,
                preprocessor=None,
                tokenizer=None,
                analyzer='word',
                stop_words=None,
                smooth_idf=True,
                sublinear_tf=True,
                ngram_range=(2, 2),
                max_df=0.75,
                min_df=1
            )
            x_train_tfidf = tfidf_vec.fit_transform(subset_train['desc_nohs'], subset_train['hs'])
            x_test_tfidf = tfidf_vec.transform(subset_test['desc_nohs'])

            clf = ComplementNB(alpha=1, norm=True)
            clf.fit(x_train_tfidf, subset_train['hs'])
            hs4_preds = clf.predict(x_test_tfidf)
            acc = accuracy_score(subset_test['hs'], hs4_preds)
            print("HS2 Code {} classifier accuracy: {}".format(hs2_code, acc))

            self.models[hs2_code] = [clf, tfidf_vec]
            self.results[hs2_code] = [hs4_preds, subset_test.index, acc]

    @staticmethod
    def gen_prediction_col(some_code_array, results_dict):
        """
        Generates a new pd.Series object by aligning the index of predictions from many different classifiers.

        Args:
            some_code_array: intended to be either MasterDataFrame.test['hs2pred'] or MasterDataFrame.test['hs4pred']
            results_dict: intended to be MasterModel.results

        Returns:
            pd.Series object holding predictions with index matching MasterDataFrame.test
        """
        print("Generating prediction column...")
        prediction_col = pd.Series()
        for code in set(some_code_array):
            pred_subset = pd.Series(results_dict[code][0], index=results_dict[code][1])
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

        test_data['hs2pred'] = MasterModel.gen_prediction_col(level0_preds, self.results)
        print("New column generated for df.test: \'hs2pred\'")

        for hs2_code in set(test_data['hs2pred']):
            subset = test_data.query("hs2pred == \'{}\'".format(hs2_code))
            x_test_tfidf = self.models[hs2_code][1].transform(subset['desc_nohs'])
            hs4_preds = self.models[hs2_code][0].predict(x_test_tfidf)
            acc = accuracy_score(subset['hs'], hs4_preds)
            print("HS2 Code {} classifier accuracy: {}".format(hs2_code, acc))

            self.results[hs2_code] = [hs4_preds, subset.index, acc]

        test_data['hs4pred'] = MasterModel.gen_prediction_col(test_data['hs2pred'], self.results)
        live_acc = accuracy_score(test_data['hs'], test_data['hs4pred'])

        self.results['live'] = live_acc
        print("Live test accuracy: ", model.results['live'])

    def base_test(self, test_data):
        for metacode in set(test_data['metacode']):
            subset = test_data.query("metacode == \'{}\'".format(metacode))
            x_test_tfidf = self.models[metacode][1].transform(subset['desc_nohs'])
            hs2_preds = self.models[metacode][0].predict(x_test_tfidf)
            acc = accuracy_score(subset['hs2'], hs2_preds)
            print("General Code {} classifier accuracy: {}".format(metacode, acc))

            self.results[metacode] = [hs2_preds, subset.index, acc]

        test_data['hs2pred'] = MasterModel.gen_prediction_col(test_data['metacode'], self.results)
        print("New column generated for df.test: \'hs2pred\'")

        for hs2_code in set(test_data['hs2pred']):
            subset = test_data.query("hs2pred == \'{}\'".format(hs2_code))
            x_test_tfidf = self.models[hs2_code][1].transform(subset['desc_nohs'])
            hs4_preds = self.models[hs2_code][0].predict(x_test_tfidf)
            acc = accuracy_score(subset['hs'], hs4_preds)
            print("HS2 Code {} classifier accuracy: {}".format(hs2_code, acc))

            self.results[hs2_code] = [hs4_preds, subset.index, acc]

        test_data['hs4pred'] = MasterModel.gen_prediction_col(test_data['hs2pred'], self.results)
        base_acc = accuracy_score(test_data['hs'], test_data['hs4pred'])

        self.results['base'] = base_acc
        print("Base test accuracy: ", self.results['base'])

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
            hs4_preds = self.models[hs2_code][0].predict(x_test_tfidf)
            self.results[hs2_code] = [hs4_preds, subset.index]

        live_data['hs4pred'] = MasterModel.gen_prediction_col(live_data['hs2pred'], self.results)
        print("New column generated for df: \'hs4pred\'")

        return x_tfidf_level0


##
# --- start data management ---
hs = MasterHS.construct()
df = MasterDataFrame.construct(hs.counts)
df.add_hs2()
df.add_desc_cat()
df.master['desc_nohs'] = df.master.apply(
    lambda row: df.add_desc_nohs(row['hs'], row['desc_cat'], alpha_only=False),
    axis=1
)
print("New column generated for df.master: 'desc_nohs'")
# hs.define_vol()
# df.add_vol(hs.vol)
df.add_metacode()
# TODO: NOTE: testing the accuracy if we drop duplicates
df.master.drop_duplicates(subset='desc_id', keep=False, inplace=True)
print("Dropped duplicates")
print("New size: ", df.master.shape)
hs.counts = np.column_stack(
    (df.master['hs'].value_counts().index.values,
     df.master['hs'].value_counts().values)
)
print("hs.counts recalculated as actual sample counts.")
# --- end data management ---

# --- start model development ---
df.get_train_test(hs.counts)
model = MasterModel()
model.level0_classification(
    train_data=df.train,
    test_data=df.test
)
model.level1_classification(
    train_data=df.train,
    test_data=df.test,
    level1_codes=df.master['metacode']
)
model.level2_classification(
    train_data=df.train,
    test_data=df.test,
    hs2_codes=df.master['hs2']
)
print("Start of live test.")
model.live_test(
    level0_preds=df.test['metapred'],
    test_data=df.test
)
model.avg_hs2_acc(
    test_data=df.test
)
# --- end model development ---
##
# evaluating the model against its base, ie no level0
model.base_test(
    test_data=df.test
)
##
# model persistence
model.persist(name='your_name_here')

##
# TODO: implement this as a method in MasterModel
# Incorporating confidence into tree method
# calculating c^ for all rows
# adam note: REMEMBER to use df_test for this
# c^ is basically the final 'probability' that is calculated for all classes. c^ = argmax{tfidf (dot) feat_logprob}
# if we look at the distribution of c^ for correct and incorrectly classified documents, we might be able to discard
# certain predictions if they fall outside of a reasonable confidence window.
# from ComplementNB() source code, to do this, we must multiply tfidf by feat_logprob
# the way CNB is implemented, we end up looking for argmax{sum(tfidf * feat_logprob)} anyway
# x_tfidf_pd has the tfidf values that we want. Can be accessed via its index, which is local (.loc)
# feat_logprob has the feature probabilities that we want. Can be accessed via its index, which is 'hs2' kinda
# can iterate this process with a list of tuples ?: [(df_train.index value, hs2-digit code), (), ()]
# for each row of x_tfidf_pd, we need to multiply all of its values against their corresponding row (key = HS2)
#   in feat_logprob.
# the relative index of df_test matches with the relative index of

##
# CALCULATING C_HAT WITHOUT TRANSFORMING SPARSE MATRIX
# working with test_x_tfidf
flp = clf_meta.feature_log_prob_

print("Generating class dictionary...")
class_dict = {}
for x in range(len(clf_meta.classes_)):
    class_dict.update({clf_meta.classes_[x]: x})

print("Calculating c_hat...")
chat_list = []
for x in range(test_x_tfidf.shape[0]):
    hs_key = df_test['metapred'].iloc[x]
    flp_key = class_dict[hs_key]
    c_hat = test_x_tfidf[x, :].dot(flp[flp_key, :])
    chat_list.append(c_hat[0])

print('Appending c_hat results to df_test...')
df_test['chat'] = pd.Series(chat_list).values
print(df_test)

print("Categorizing c_hat results...")
chat_results = {'right': [], 'wrong': []}


def chat_filter(row):
    hs2_key = row['hs2']
    meta_key = row['meta_pred']
    chat_key = row['chat']
    if hs2_key == meta_key:
        chat_results['right'].append(chat_key)
    else:
        chat_results['wrong'].append(chat_key)


print("Applying chat_filter to df_test.")
df_test.apply(lambda row: chat_filter(row), axis=1)

print("Fixing some stuff...")
while len(chat_results['wrong']) < len(chat_results['right']):
    chat_results['wrong'].append(0)

chat_results_pd = pd.DataFrame.from_dict(chat_results)
print("chat_results_pd created.")

##
# CONFIDENCE ADJUSTMENT CODE
percentile_cutoff = np.percentile(chat_results_pd['right'], 80)


def perc_adjust(row):
    c_hat_key = row['chat']
    if c_hat_key > percentile_cutoff:
        return row['desc_id']


cut_desc_ids = list(df_test.apply(lambda row: perc_adjust(row)))

##
# adam notes
"""
sparse.csr_matrix.data seems like it gives you all of the non_zero values in the matrix.
sparse.csr_matrix.count_nonzero() does what it says.
"""
##
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
