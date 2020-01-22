# adam note: '##' used for testing in pycharm
##
# built-in
import math
import re
# external
import psycopg2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score

# --- set constants ---
SAMPLE_PCT = 0.50  # percent of total obs sampled for each HS code
TRAIN_PCT = 0.70  # percent of sampled obs used for training (complement used for testing)
PERCENTILES = (20, 50, 70, 90, 99)  # used to generated volume codes predicted in level0 classification
HOSTNAME = "xxx-xx-xxx-xxx-xx.compute-1.amazonaws.com"
# ---------------------

# --- connect to dev db ---
print("Establishing connection with database...")
conn = psycopg2.connect(
    host=HOSTNAME,
    database="db",
    port=5432,
    user="user",
    password="pw"
)
cur = conn.cursor()
print("Connection established.")
# -------------------------


class MasterHS:
    """
    Initialize with MasterHS.construct(<np array>)
    """
    def __init__(self, counts):
	"""
	Holds information about the 4-digit HS codes we are trying to classify
	
	Args:
		counts (np array): nx2 with col1: str(4-digit HS code), col2: str(total # of obs)
		
	Attrs:
		counts (np array): is above
		ptiles (np array): contains percentiles of self.counts calculated using PERCENTILES
		vol (dict): with keys: str(4-digit HS code), values: int(volume code)
	"""
        self.counts = counts
        self.ptiles = None
        self.vol = None
        
    @classmethod
    def construct(cls):
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

    def define_vol(self, ptiles=PERCENTILES):
        """
		Defines a set of 'volume codes' based on self.ptiles 
        Dependent variable during level0 classification.
        Sets inner 'ptiles' attribute and instantiates inner 'vol' dict.
		
        Args:
            ptiles (array): holds desired percentiles defined in global vars.
                            Automatically set by constant 'PERCENTILES'.
        Returns:
			print
        """
		print("Defining volume codes...")
        self.ptiles = np.percentile(self.counts[:, 1].astype(int), ptiles)
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

        return print("Volume codes defined based on given percentiles.")


class MasterDataFrame:
    """
    Initialize with MasterDataFrame.construct(<MasterHS.counts>).
    """

    def __init__(self, dataframe):
	"""
	Holds a stratified random sample of container description data (and subsets thereof)
	
	Args:
		dataframe (pd.DataFrame): nx7 array passed from construction query
		
	Attrs:
		master (pd.DataFrame): see above. original copy
		train (pd.DataFrame): subset of self.master set by self.get_train_test()
		test (pd.DataFrame): subset of self.master set by self.get_train_test()
								complement of self.train
		kfolds (dict): holds k-folds of self.master
		get_k (int): returns 'k'
	"""
        self.master = dataframe
        self.train = None
        self.test = None
        self.kfolds = {}
        self.get_k = None

    @classmethod
    def construct(cls, hs_counts):
        """
        Constructs instance of MasterDataFrame from dev db query.

        Args:
            hs_counts: this should be MasterHS.counts
        Returns:
            cls: instance of the MasterDataFrame class
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
        print("Data constructed from stratified sample.")

        return cls(dataframe=pd.DataFrame.from_records(cur.fetchall(), columns=colnames))

    def add_hs2(self):
        self.master['hs2'] = self.master['hs'].str[:2]
        print("New column generated in df.master: \'hs2\'")

    def add_desc_cat(self):
        colnames = ['desc', 'port_origin', 'port_us', 'shipper', 'consignee']
        self.master.fillna('', inplace=True)
        new_col = self.master[colnames[0]].values
        for name in colnames[1:]:
            new_col += ' ' + self.master[name].values

        self.master['desc_cat'] = new_col
        print("New column generated in df.master: \'desc_cat\'")

    @staticmethod
    def add_desc_nohs(hs_code, desc, alpha_only=False):
        """
        Strips parsed HS code from description text. Potentially strips [^A-za-z].
        Passed to lambda function as an argument for df.apply().

        Args:
            hs_code: str, 4-digit HS code
            desc: str, description text (ideally an element of 'desc_cat')
            alpha_only: bool, whether or not to strip everything except alphabetic chars

        Returns:
            str: element of new col: 'desc_nohs'
        """
        pattern = r'[^\d]' + hs_code + r'([^\d]|\d{2}[^\d]|\d{4}[^\d]|\d{6}[^\d])'
        desc_nohs = re.sub(pattern, ' ', desc)
        if alpha_only:
            desc_nohs = re.sub(r'[^A-Za-z]', ' ', desc_nohs)

        return desc_nohs.lower()

    def add_vol(self, vol_dict):
        new_col = []
        for hs_code in self.master['hs']:
            new_col.append(vol_dict[hs_code])

        self.master['vol'] = new_col
        print("New column generated in df.master: \'vol\'")

    def get_train_test(self, hs_counts):
        """
        Splits MasterDataFrame into subsets for training and testing based on the provided parameters.

        Args:
            hs_counts:     array, holds each hs code and a 'count' of its total appearances in dev.zad_hs_results.
        """
        train_index = []
        for i in range(hs_counts.shape[0]):
            hs_code = hs_counts[i, 0]
            hs_count = int(hs_counts[i, 1])
            hs_obs = math.ceil(hs_count * SAMPLE_PCT * TRAIN_PCT)
            query = "hs == \'{}\'".format(hs_code)
            train_index += list(self.master.query(query).sample(n=hs_obs).index)
        train_index.sort()
        self.train = self.master.loc[train_index]
        self.test = self.master[~self.master.isin(self.train)].dropna(how='all')

        return print("Train/test split complete.")

    def get_strat_kfolds(self, folds=5, shuffle=False):
        """
        Access individual folds through:
            self.kfolds[k]            where k is an int in range(self.get_k)
            self.kfolds[k][0]        use 0 for X data and 1 for y data
            self.kfolds[k][0][0]    use 0 for train_index and 1 for test_index

        Args:
            folds: int, number of folds
            shuffle: bool, shuffles data before splitting
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
        self.models = {}
        self.results = {}

    def level0_classification(self, train_data, test_data):
        x_train = train_data['desc_nohs']
        y_train = train_data['vol']
        x_test = test_data['desc_nohs']
        y_test = test_data['vol']
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
            max_df=0.20,
            min_df=1
        )
        x_train_tfidf = tfidf_vec.fit_transform(x_train, y_train)
        x_test_tfidf = tfidf_vec.transform(x_test)

        clf = ComplementNB(alpha=1, norm=True)
        clf.fit(x_train_tfidf, y_train)
        volume_code_preds = clf.predict(x_test_tfidf)
        acc = accuracy_score(y_test, volume_code_preds)
        print("level0 classification accuracy: {}".format(acc))

        test_data['volpred'] = volume_code_preds
        print("New column generated in df.test: \'volpred\'")
        self.models['level0'] = [clf, tfidf_vec]
        self.results['level0'] = [volume_code_preds, acc]

    def level1_classification(self, volume_codes, train_data, test_data):
        for volume_code in set(volume_codes):
            query = "vol == \'{}\'".format(volume_code)
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
                max_df=0.50,
                min_df=1
            )
            x_train_tfidf = tfidf_vec.fit_transform(subset_train['desc_nohs'], subset_train['hs2'])
            x_test_tfidf = tfidf_vec.transform(subset_test['desc_nohs'])

            clf = ComplementNB(alpha=1, norm=True)
            clf.fit(x_train_tfidf, subset_train['hs2'])
            hs2_preds = clf.predict(x_test_tfidf)
            acc = accuracy_score(subset_test['hs2'], hs2_preds)
            print("Volume Code {} classifier accuracy: {}".format(volume_code, acc))

            self.models[volume_code] = [clf, tfidf_vec]
            self.results[volume_code] = [hs2_preds, subset_test.index, acc]

    def level2_classification(self, hs2_codes, train_data, test_data):
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
                max_df=0.50,
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

    def live_test(self, volume_preds, test_data):
        for volume_code in set(volume_preds):
            subset = test_data.query("volpred == \'{}\'".format(volume_code))
            x_test_tfidf = self.models[volume_code][1].transform(subset['desc_nohs'])
            hs2_preds = self.models[volume_code][0].predict(x_test_tfidf)
            acc = accuracy_score(subset['hs2'], hs2_preds)
            print("Volume Code {} classifier accuracy: {}".format(volume_code, acc))

            self.results[volume_code] = [hs2_preds, subset.index, acc]

        test_data['hs2pred'] = MasterModel.gen_prediction_col(volume_preds, self.results)
        print("New column generated in df.test: \'hs2pred\'")

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


##
# --- start data management ---
hs = MasterHS.construct()
df = MasterDataFrame.construct(hs.counts)
df.add_hs2()
df.add_desc_cat()
df.master['desc_nohs'] = df.master.apply(
    lambda row: df.add_desc_nohs(row['hs'], row['desc_cat'], alpha_only=True),
    axis=1
)
print("New column generated in df.master: 'desc_nohs'")
hs.define_vol()
df.add_vol(hs.vol)
# --- end data management ---

# --- start model development ---
# TODO: with current percentiles, level0 classification is low at 80%
#       try decreasing the number of ptiles instead. (70, 99) or something like that.
#       Really we need to isolate those huge HS codes that throw everything else off.
#       Maybe also try (10, 90, 99). The middle is pretty good but it's the outliers...
df.get_train_test(hs.counts)
model = MasterModel()
model.level0_classification(
    train_data=df.train,
    test_data=df.test
)
model.level1_classification(
    train_data=df.train,
    test_data=df.test,
    volume_codes=df.master['vol']
)
model.level2_classification(
    train_data=df.train,
    test_data=df.test,
    hs2_codes=df.master['hs2']
)
model.live_test(
    volume_preds=df.test['volpred'],
    test_data=df.test
)
print(model.results['live'])
# --- end model development ---

##
# computes accuracy for the base model ignoring volume code predictions
# TODO: wrap this up in the MasterModel class as a new method or maybe integrate into live_test().
for volume_code in set(df.test['vol']):
    subset = df.test.query("vol == \'{}\'".format(volume_code))
    x_test_tfidf = model.models[volume_code][1].transform(subset['desc_nohs'])
    hs2_preds = model.models[volume_code][0].predict(x_test_tfidf)
    acc = accuracy_score(subset['hs2'], hs2_preds)
    print("Volume Code {} classifier accuracy: {}".format(volume_code, acc))

    model.results[volume_code] = [hs2_preds, subset.index, acc]

df.test['hs2pred'] = MasterModel.gen_prediction_col(df.test['vol'], model.results)
print("New column generated in df.test: \'hs2pred\'")

for hs2_code in set(df.test['hs2pred']):
    subset = df.test.query("hs2pred == \'{}\'".format(hs2_code))
    x_test_tfidf = model.models[hs2_code][1].transform(subset['desc_nohs'])
    hs4_preds = model.models[hs2_code][0].predict(x_test_tfidf)
    acc = accuracy_score(subset['hs'], hs4_preds)
    print("HS2 Code {} classifier accuracy: {}".format(hs2_code, acc))

    model.results[hs2_code] = [hs4_preds, subset.index, acc]

df.test['hs4pred'] = MasterModel.gen_prediction_col(df.test['hs2pred'], model.results)
live_acc = accuracy_score(df.test['hs'], df.test['hs4pred'])

model.results['live'] = live_acc
print(model.results['live'])

##
# return average accuracy of hs2 models
total = 0
for hs2 in set(df.test['hs2pred']):
    total += model.results[hs2][1]
print(total / len(set(df.test['hs2pred'])))
