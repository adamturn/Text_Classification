#!/usr/bin/python3
# Python 3.6.8

# standard library
import re
import math
import random
from time import process_time
# third-party
import joblib
import psycopg2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
# third-party modified
from sklearn.naive_bayes_mod import ComplementNB
# first-party
import conndb

# adam note:
#     check path to properties file
STRFMT0 = " "*2 + "> "


class LiveData(object):
    """Derived class that holds live data from the db.

    Attributes:
        og (pd.DataFrame): original data pulled from db
        df (pd.DataFrame): clean data passed from Wrangler
        train (pd.DataFrame): training subset
        test (pd.DataFrame): testing subset; complement of train relative to df
        counts (pd.Series): series with code along the axis and counts as values
    """

    def __init__(self):
        propspath = "/path/to/config_file"
        self.con = conndb.create_connection(propspath)
        self.og = None
        self.df = None
        self.train = None
        self.test = None
        self.counts = None

    @staticmethod
    def create_records(cursor, batch_size):
        print(f"{STRFMT0}Fetching data records...")
        start_time = process_time()
        data_records = list()
        data_records.extend(cursor.fetchmany(batch_size))
        while data_records:
            batch = cursor.fetchmany(batch_size)
            if batch:
                data_records.extend(batch)
            else:
                break
        print(f"{STRFMT0}Fetch time:", process_time() - start_time)

        return data_records

    def __backup_og(self, data_records, filepath, colnames):
        """Creates original data atrribute and writes itself to disk.

        Args:
            data_records (list[tuple]): holds data from db
            filepath (str): where to save the original data
            colnames (list[str]): column names for the original data
        """
        self.og = pd.DataFrame.from_records(data_records, columns=colnames)
        print(self.og)
        print("Writing data to csv at:\n", filepath)
        self.og.to_csv(filepath, index=False)
        return self

    def __compute_totals(self, pandas_df, partition_on="hs4"):
        """Counts the total number of observations per element of the set that is partitioned on.

        Args:
            pandas_df (pd.DataFrame): dataset to count
            partition_on (str): column name used as index for count
        """
        # returns pd.Series having counts as values with hs4 codes as index
        start_time = process_time()
        self.counts = pandas_df[partition_on].value_counts(
            normalize=False, sort=True, ascending=False, bins=None, dropna=True
        )
        print(f"{STRFMT0}Shape: {self.df.shape}")
        print(f"{STRFMT0}Compute totals speed:", process_time() - start_time)
        return self

    def __dump_record_id(self, filepath):
        print("Fetching max record id...")
        client_cursor = self.con.cursor()
        query = "select max(record_id) from cbp.import_full;"
        client_cursor.execute(query)
        latest_record_id = client_cursor.fetchall()
        print(f"{STRFMT0}Latest:", latest_record_id)
        print(f"{STRFMT0}Dumping to {filepath}...")
        joblib.dump(latest_record_id, filepath, compress=0)
        print(f"{STRFMT0}Done.")
        return self

    def query_latest_data(self, limit=True, min_record_id=False):
        """Query the db and get live data. Work in Progress."""
        # get the min record id
        record_id_path = "/home/adam/text_class/project_cache/latest_id.joblib"
        if not min_record_id:
            try:
                min_record_id = joblib.load(record_id_path)
            except FileNotFoundError as e:
                print(f"{e} -- The local directory '/project_cache' might not exist.")

        # get the data
        print("Server cursor opened!")
        ss_cursor = self.con.cursor("cnb_create_records")
        print(f"{STRFMT0}Querying the latest data...")
        query = "select record_id::text, " \
                "description_id::text, " \
                "description::text, " \
                "fulltext_description::text, " \
                "port_us::text, " \
                "port_origin::text, " \
                "shipper::text, " \
                "consignee::text " \
                "from cbp.import_full " \
                f"where record_id > {min_record_id}"
        if limit:
            query += f" limit 100;"
        else:
            query += ";"
        ss_cursor.execute(query)
        data_records = LiveData.create_records(cursor=ss_cursor, batch_size=100_000)
        ss_cursor.close()
        print(f"{STRFMT0}Server cursor closed!")
        colnames = ["id", "desc_id", "desc", "ftdesc", "port_us", "port_og", "ship", "cons"]
        datapath = "/home/adam/pycharm_remote/branches/latestdata.csv"
        self.__dump_record_id(record_id_path)
        self.__backup_og(data_records, datapath, colnames)
        return self

    def build_sampling_table(self, sample_pct=0.10):
        """Sample some of the latest import data from our db!

        Args:
            sample_pct (float) = percent of latest obs to sample for each hs4 code
        """
        client_cur = self.con.cursor()
        print("Created client cursor.")

        print(f"{STRFMT0}Getting HS4 counts...")
        query = "select hs, count(hs) as hs_count " \
                "from dev.zad_hs_results " \
                "group by hs " \
                ";"
        client_cur.execute(query)
        hs4_code_counts = pd.DataFrame(client_cur.fetchall(), columns=["hs4", "count"])
        # drop all hs4 codes that don't have at least 4 observations
        low_volume = hs4_code_counts[hs4_code_counts["count"] < 4]
        print(f"{STRFMT0}Removing low volume HS4 codes...", low_volume["hs4"])
        hs4_code_counts = hs4_code_counts[hs4_code_counts["count"] >= 4]
        # worry about that later
        hs4_code_counts = hs4_code_counts.to_records(index=False)
        print(f"{STRFMT0}Initializing sampling table...")
        query = "drop table if exists dev.zad_construct_sample; " \
                "create table dev.zad_construct_sample as (" \
                "select * from dev.zad_construct_full limit 0" \
                ");"
        client_cur.execute(query)

        print(f"{STRFMT0}Building...")
        start_time = process_time()
        sql_payload = str()
        for hs4_code_count in hs4_code_counts:
            total_sample_obs = math.floor(int(hs4_code_count[1]) * sample_pct)
            query = "insert into dev.zad_construct_sample (" \
                    "select * from dev.zad_construct_full " \
                    f"where hs = \'{hs4_code_count[0]}\' " \
                    f"limit {total_sample_obs}" \
                    "); "
            sql_payload += query
        client_cur.execute(sql_payload)
        print(f"{STRFMT0}Done in {process_time() - start_time}!")
        client_cur.close()
        print(f"{STRFMT0}Client cursor closed!")

        ss_cursor_name = "cnb_build_sampling_table"
        ss_cursor = self.con.cursor(ss_cursor_name)
        print("Server cursor opened!")
        print(f"{STRFMT0}Executing...")
        start_time = process_time()
        query = "select record_id::text, " \
                "description_id::text, " \
                "description::text, " \
                "fulltext_description::text, " \
                "port_us::text, " \
                "port_origin::text, " \
                "shipper::text, " \
                "consignee::text, " \
                "hs::text " \
                "from dev.zad_construct_sample" \
                ";"
        ss_cursor.execute(query)
        print(f"{STRFMT0}Server cursor execute time:", process_time() - start_time)
                
        data_records = LiveData.create_records(cursor=ss_cursor, batch_size=100_000)
        ss_cursor.close()
        print(f"{STRFMT0}Server cursor closed!")
        self.con.commit()
        print("LiveData connection commit!")
        self.con.close()
        print("LiveData connection closed!")
        colnames = ["id", "desc_id", "desc", "ftdesc", "port_us", "port_og", "ship", "cons", "hs4"]
        datapath = "/home/adam/pycharm_remote/branches/livedata.csv"
        self.__backup_og(data_records, datapath, colnames)
        return self

    def read_csv(self, filepath):
        """Reads old LiveData.og from disk."""
        print("Reading live data...")
        start_time = process_time()
        # dtypes are all str but ship/cons sometimes have NaN
        og = pd.read_csv(filepath, dtype=str)
        # make NaN 'empty' str for later
        og = og.replace(np.nan, 'empty', regex=True)
        self.og = og
        print(f"{STRFMT0}Read time:", process_time() - start_time)
        return self

    def clean_for_training(self):
        """Cleaning method that performs all common wrangling methods on training data.

        Returns modified self.df attribute equal to a clean dataframe for experimentation.
        """
        print("Cleaning data...")

        print(f"{STRFMT0}Replacing nans...")
        self.df = self.og.replace(np.nan, "empty", regex=True)

        print(f"{STRFMT0}Dropping duplicates...")
        self.df.drop_duplicates(subset=["desc", "ship", "cons"], keep='last', inplace=True)

        # drop all hs4 codes with less than 4 observations
        self.__compute_totals(pandas_df=self.df, partition_on="hs4")
        low_vol = list(self.counts[self.counts < 4].index)
        print(f"{STRFMT0}Low volume:\n\n", pd.Series(low_vol))
        self.df = self.df[~self.df["hs4"].isin(low_vol)]

        # let Wrangler do the heavy lifting
        self.df = Wrangler.format_raw_text(self.df)
        self.df = Wrangler.crack_hs_code(self.df)
        self.df = self.df[["id", "desc+", "ftdesc+", "hs4", "hs2", "hs1"]]

        return self


    def strat_train_test2(self, train_pct=0.70, seed=42):
        print("StratTrainTest2 executing momentarily!")
        random.seed(seed)
        self.__compute_totals(pandas_df=self.df, partition_on="hs4")

        start_time = process_time()

        training_indices = list()
        for hs4 in set(self.df["hs4"]):
            total_training_obs = math.floor(self.counts[hs4] * train_pct)
            subset_indices = random.sample(
                population=self.df["hs4"].index[self.df["hs4"] == hs4].tolist(),
                k=total_training_obs
            )
            training_indices.extend(subset_indices)

        self.train = self.df.loc[training_indices]
        df_idx = self.df.index
        training_indices = set(training_indices)
        testing_indices = [idx for idx in df_idx if idx not in training_indices]
        self.test = self.df.loc[testing_indices]
        # self.test = self.df[~self.df.isin(self.train)].dropna(how="all")
        print(f"{STRFMT0}STT2 speed:", process_time() - start_time)
        print(f"{STRFMT0}Train split: {self.train.shape}")
        print(f"{STRFMT0}Test split: {self.test.shape}")
        return self


class Wrangler(object):
    """Makes the data work."""

    @staticmethod
    def __concat_text(arglist):
        """Combines text data args provided by arglist."""
        text = str()
        for arg in arglist:
            text += " " + arg
        return text

    @staticmethod
    def __format_fulltext(fulltext_record):
        # create list of fulltext vectors
        vectors = fulltext_record.split(" ")
        # form list of tuples: (index, fulltext vector)
        vectors = [(int(vec[-1]), vec[:-1]) for vec in vectors]
        # sort list (asc) of tuples by tuple[0]: index
        vectors.sort(key=lambda vec: vec[0])
        # return formatted fulltext vectors as str
        return " ".join([vec[1] for vec in vectors])

    @staticmethod
    def crack_hs_code(pandas_df):
        """Creates two additional columns by breaking down the HS4 code."""
        df = pandas_df
        df["hs2"] = df["hs4"].str[:2]
        df["hs1"] = df["hs4"].str[0]
        return df

    @staticmethod
    def format_raw_text(pandas_df):
        """Performs all standard cleaning methods on the data.

        Requirements: ['desc', 'ftdesc', 'port_us, 'port_og', 'ship', 'cons'] columns
        """
        df = pandas_df

        # format the fulltext description
        print(f"{STRFMT0}Format fulltext descriptions...")
        df["ftdesc"] = df["ftdesc"].apply(Wrangler.__format_fulltext)

        # we need to first standardize the ports/shipper/consignee cols
        # for example, a shipper record with value "Crude Company Name" becomes: "ship_CrudeCompanyName"
        print(f"{STRFMT0}Format extra text columns...")
        cols = ["port_us", "port_og", "ship", "cons"]
        df_cols = list()
        for col in cols:
            df[col] = Wrangler.__concat_text([col + "_", df[col].str.replace(" ", "")])
            df_cols.append(df[col])

        # now we need to merge all formatted cols into ftdesc -> ftdesc+ and desc -> desc+
        print(f"{STRFMT0}Creating desc+ and ftdesc+ columns...")
        df_cols.insert(0, df["ftdesc"])
        df["ftdesc+"] = Wrangler.__concat_text(df_cols)
        df_cols.pop(0)
        df_cols.insert(0, df["desc"])
        df["desc+"] = Wrangler.__concat_text(df_cols)

        return df

    @staticmethod
    def remove_raw_hs(pandas_df, desc_colname, hs4_colname="hs4", sdw_only=False):
        """Removes raw HS4 code from a description column.

        Args:
            pandas_df (pd.DataFrame): to operate on
            desc_colname (str): the column name of the description we want to work with
            hs4_colname (str): name of column that has the 4-digit HTS codes
            sdw_only (bool): if True, also strip everything except whitespace, digit, word chars
        """
        df = pandas_df
        desc_nohs = pd.Series()
        for hs4 in set(df[hs4_colname]):
            hs4_pattern = re.compile(r"[^\d]" + hs4 + r"([^\d]|\d{2}[^\d]|\d{4}[^\d]|\d{6}[^\d])")
            subset = df[df[hs4_colname == hs4]][desc_colname]
            # subset = df.query(hs4_colname + f" == \'{hs4}\'")[desc_colname]
            subset = subset.str.replace(hs4_pattern, "")
            if sdw_only:
                sdw_pattern = re.compile(r"[^\s\d\w]")
                subset = subset.str.replace(sdw_pattern, " ")
            desc_nohs = desc_nohs.append(subset)
        return desc_nohs.sort_index().str.lower()

    @staticmethod
    def unpack_results(results, base_colname, cvals=False):
        """Unpack those pesky result lists!

        Args:
            results (pd.Series): size n x 1 where each record must be list with length 3
            base_colname (str): column name used as a prefix for the unpacked columns
            cvals (bool): are these c values? if so, change the basename to reflect that.
        Returns: pd.DataFrame of size n x 3 with unpacked result vectors for rows
        """
        if cvals:
            base_colname += "c"
        buffer = pd.DataFrame()
        buffer[[base_colname + "_3", base_colname + "_2", base_colname + "_1"]] = pd.DataFrame(
            results.to_list(), index=results.index
        )
        return buffer

    @staticmethod
    def vectorize(xtrain, xtest, stopwords=None, ngrams=(2, 2), maxdf=1.00):
        """Vectorize your training & testing data in one step with this handy method.

        Turns nx1 rows of text data into nxf array of features where f is the
        total number of features identified in the text data by the vectorizer.

        Returns nxf np.array with tfidf vectors as records which
        can be used as input to our text classification algorithm.
        """
        print(f"{STRFMT0}Transforming...")
        tfidf_vec = TfidfVectorizer(
            strip_accents='ascii', lowercase=True, preprocessor=None, analyzer='word',
            stop_words=stopwords,
            smooth_idf=True, 
            sublinear_tf=True,
            ngram_range=ngrams,
            max_df=maxdf, min_df=1
        )
        train_tfidf = tfidf_vec.fit_transform(xtrain)
        test_tfidf = tfidf_vec.transform(xtest)
        return tfidf_vec, train_tfidf, test_tfidf

    @staticmethod
    def export_model(model, model_name):
        print("Exporting model...")
        export_path = f"/home/adam/text_class/models/{model_name}.joblib"
        print(f"{STRFMT0}Compressing...")
        joblib.dump(model, export_path, compress=1, protocol=4)
        print(f"{STRFMT0}BayesNet object now persists at:\n\t{export_path}")
        return model

    @staticmethod
    def load_model(model_path):
        print("Loading model...")
        model = joblib.load(filename=model_path)
        if isinstance(model, BayesNet):
            return model
        else:
            raise TypeError("The loaded joblib file does not evaluate to a BayesNet object.")


class BayesNet(object):
    """Multilayer Bayesian Network

    Attributes:
        graph (dict): hash table with
            keys('node_root', 'node_' + any hs1, hs2, h4 code)
            values({'vec': tfidf_vec, 'clf': clf})
        results (dict): hash table with
            keys(all nodenames used above, layer_hs1, layer_hs2)
            values({"acc": acc, "results": hspreds, "results_c": hspreds_c})
    """

    def __init__(self):
        self.graph = dict()
        self.results = dict()

    def train_node(self, xtrain, ytrain, xtest, ytest, nodename, stopwords=None, ngrams=(2, 2), maxdf=1.0):
        print("Training node:", nodename)
        tfidf_vec, train_tfidf, test_tfidf = Wrangler.vectorize(xtrain, xtest, stopwords, ngrams, maxdf)
        clf = ComplementNB(alpha=0.05, norm=False)
        print(f"{STRFMT0}Learning...")
        clf.fit(train_tfidf, ytrain)

        print(f"{STRFMT0}Predicting...")
        hspreds, hspreds_c = clf.predict_top3(test_tfidf)
        acc = accuracy_score(ytest, hspreds[:, -1])
        print(f"{STRFMT0}{nodename} node accuracy: {acc}")

        hspreds = pd.Series(hspreds.tolist(), index=ytest.index)
        hspreds_c = pd.Series(hspreds_c.tolist(), index=ytest.index)

        nodename = "node_" + nodename
        self.graph[nodename] = {"vec": tfidf_vec, "clf": clf}
        self.results[nodename] = {"acc": acc, "hspreds": hspreds, "hspreds_c": hspreds_c}
        return self.results[nodename]

    def train_layer(self, train_data, test_data, desc_type, y_in_type, y_out_type):
        print(f"Training layer with input: {desc_type}, predicting: {y_out_type}")
        # initialize data structures to hold results
        layer_results = pd.Series()
        layer_results_c = pd.Series()
        # loop through each y in the input set & train & node for each
        for y_in in set(train_data[y_in_type]):
            train_subset = train_data[train_data[y_in_type] == y_in]
            xtrain = train_subset[desc_type]
            ytrain = train_subset[y_out_type]

            test_subset = test_data[test_data[y_in_type] == y_in]
            xtest = test_subset[desc_type]
            ytest = test_subset[y_out_type]

            # node_results will be self.results[nodename]
            node_results = self.train_node(
                xtrain=xtrain,
                ytrain=ytrain,
                xtest=xtest,
                ytest=ytest,
                nodename=y_in,
                stopwords=None,
                ngrams=(2, 2),
                maxdf=1.0
            )
            layer_results = layer_results.append(node_results["hspreds"])
            layer_results_c = layer_results_c.append(node_results["hspreds_c"])
        # sort results by index
        layer_results.sort_index(inplace=True)
        layer_results_c.sort_index(inplace=True)
        # unpack results list into 3 cols, keeping track of the indices
        # our layer results will become n x 3 pd.DataFrames
        print(f"Unpacking {y_out_type} results...")
        layer_results = Wrangler.unpack_results(layer_results, y_out_type)
        layer_results_c = Wrangler.unpack_results(layer_results_c, y_out_type, cvals=True)
        acc = accuracy_score(test_data[y_out_type], layer_results.iloc[:, -1])
        print(f"{STRFMT0}{y_out_type} layer accuracy: {acc}")
        # access specific results with relevant prefix, inner dict with keys: 'acc' | 'results' | 'results_c'
        # example layer level: BayesNet.results['layer_hs2']['acc']
        # example node level: BayesNet.results['node_39']['results']
        self.results["layer_" + y_out_type] = {"acc": acc, "results": layer_results, "results_c": layer_results_c}
        return self.results["layer_" + y_out_type]

    def train_graph(self, train_data, test_data, desc_type="desc+", root_y="hs1"):
        print("Training graph with input type:", desc_type)
        # train root node (predicts hs1 from xdata)
        xtrain = train_data[desc_type]
        ytrain = train_data[root_y]
        xtest = test_data[desc_type]
        ytest = test_data[root_y]
        self.train_node(xtrain, ytrain, xtest, ytest, nodename="root")

        # train layer 1 (predicts hs2 from hs1)
        self.train_layer(train_data, test_data, desc_type, y_in_type="hs1", y_out_type="hs2")

        # train layer 2 (predicts hs4 from hs2)
        self.train_layer(train_data, test_data, desc_type, y_in_type="hs2", y_out_type="hs4")

        return self

    def execute_root(self, data, desc_type, class_type="hs1"):
        """Basic method that executes the graph's root node.

        Args:
            data (pd.DataFrame): contains clean description column
            desc_type (str): column with the xdata to classify
            class_type (str): this is the class_type that root is predicting
 
        Returns: tuple(pd.Series[class3, class2, class1], 
                       pd.Series[cvalue3, cvalue3, cvalue1])
        """
        print(f"{STRFMT0}Executing root node!")
        x_tfidf = self.graph["node_root"]["vec"].transform(data[desc_type])
        root_results, root_results_c = self.graph["node_root"]["clf"].predict_top3(x_tfidf)
        root_results = pd.Series(root_results.tolist(), index=data.index)
        root_results_c = pd.Series(root_results_c.tolist(), index=data.index)

        print(f"{STRFMT0}Unpacking root node results...")
        root_results = Wrangler.unpack_results(root_results, class_type)
        root_results_c = Wrangler.unpack_results(root_results_c, class_type, cvals=True)

        return root_results, root_results_c

    def execute_layer(self, data, desc_type, y_in_type, y_out_type):
        print(f"{STRFMT0}Using {y_in_type} results to predict {y_out_type}!")
        layer_results = pd.Series()
        layer_results_c = pd.Series()
        for nodename in set(data[y_in_type]):
            subset = data[data[y_in_type] == nodename]
            x_subset = subset[desc_type]

            node = self.graph["node_" + nodename]
            x_tfidf = node["vec"].transform(x_subset)
            node_results, node_results_c = node["clf"].predict_top3(x_tfidf)

            node_results = pd.Series(node_results.tolist(), index=x_subset.index)
            node_results_c = pd.Series(node_results_c.tolist(), index=x_subset.index)

            layer_results = layer_results.append(node_results)
            layer_results_c = layer_results_c.append(node_results_c)

        layer_results.sort_index(inplace=True)
        layer_results_c.sort_index(inplace=True)
        print(f"{STRFMT0}Unpacking {y_out_type} results...")
        layer_results = Wrangler.unpack_results(layer_results, y_out_type)
        layer_results_c = Wrangler.unpack_results(layer_results_c, y_out_type, cvals=True)

        return layer_results, layer_results_c

    def execute_graph(self, data, desc_type="desc+", test=False):
        print("Directing graph...")
        # EXECUTE ROOT NODE
        root_results, root_results_c = self.execute_root(data, desc_type)
        root_results = root_results.join(root_results_c, how="left")
        data = data.join(root_results, how="left")

        # EXECUTE LAYER 1
        layer_results1, layer_results1_c = self.execute_layer(data, desc_type, y_in_type="hs1_1", y_out_type="hs2")
        layer_results1 = layer_results1.join(layer_results1_c, how="left")
        data = data.join(layer_results1, how="left")

        # EXECUTE LAYER 2
        layer_results2, layer_results2_c = self.execute_layer(data, desc_type, y_in_type="hs2_1", y_out_type="hs4")
        layer_results2 = layer_results2.join(layer_results2_c, how="left")
        data = data.join(layer_results2, how="left")

        if test:
            acc = accuracy_score(data["hs1"], data["hs1_1"])
            print("HS1 accuracy:", acc)
            acc = accuracy_score(data["hs2"], data["hs2_1"])
            print("HS2 accuracy:", acc)
            acc = accuracy_score(data["hs4"], data["hs4_1"])
            print("HS4 accuracy:", acc)
            print("Exporting data...")
            data.to_csv("/home/adam/text_class/data/livedata_results.csv")

        return data


def traintest():
    # samplepct = 0.10
    # data = LiveData().build_sampling_table(sample_pct=samplepct)
    data = LiveData().read_csv("/home/adam/pycharm_remote/branches/traindata_10.csv")

    data.clean_for_training()
    data.strat_train_test2(train_pct=0.70)
    model = BayesNet().train_graph(data.train, data.test, desc_type="desc+", root_y="hs1")
    model.execute_graph(data.test, "desc+", test=True)

    modelname = "cnbv3-0-10"
    Wrangler.export_model(model, modelname)
    return model


def main():
    traintest()
    return None


if __name__ == "__main__":
    main()
