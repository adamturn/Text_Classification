# Python 3.6.8

# adam note:
#     check path to properties file

# debug
import pdb
# standard library
import re
import math
from time import process_time
# third-party
import psycopg2
import pandas as pd
from joblib import dump
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
# third-party modified
from sklearn.naive_bayes_mod import ComplementNB
# first-party
import conndb


class DBInterface(object):
    """Base class that interfaces with the db.

    Attributes:
        con (psycopg2 connection object): establishes connection with the db
        cur (psycopg2 cursor object): interacts with the db
    """
    def __init__(self, connection, ss_cursor):
        """
        Args:
            connection (psycopg2 cursor object): passed from derived class constructor
        """
        self.con = connection
        self.cur = connection.cursor(ss_cursor)


class LiveData(DBInterface):
    """Derived class that holds live data from the db.

    Attributes:
        og (pd.DataFrame): original data pulled from db
        df (pd.DataFrame): clean data passed from Wrangler
        train (pd.DataFrame): training subset
        test (pd.DataFrame): testing subset; complement of train relative to df
        counts (pd.Series): series with code along the axis and counts as values
    """

    def __init__(self, connection, ss_cursor):
        """Initialize with LiveData.construct() class method.

        Args:
            connection (psycopg2 connection object): passed from LiveData.connect() class method
        """
        super().__init__(connection, ss_cursor)
        self.og = None
        self.df = None
        self.train = None
        self.test = None
        self.counts = None

    @classmethod
    def connect(cls, cursor_name=None):
        """Connects to the db and returns a connected LiveData object."""
        props = conndb.parse_props("/home/config_files/config_dev.properties")

        print("Connecting to db...")
        conn = psycopg2.connect(
            host=props['db_host'],
            database=props['db_name'],
            port=props['db_port'],
            user=props['db_user'],
            password=props['db_password']
        )
        print("{}> Connection established!".format(' ' * 2))

        return cls(connection=conn, ss_cursor=cursor_name)

    def compute_totals(self, pandas_df, partition_on="hs4"):
        """Counts the total number of observations per element of the set that is partitioned on.

        Args:
            pandas_df (pd.DataFrame): dataset to count
            partition_on (str): column name used as index for count
        """
        # returns pd.Series having counts as values with hs4 codes as index
        print("Computing totals...")
        self.counts = pandas_df[partition_on].value_counts(
            normalize=False, sort=True, ascending=False, bins=None, dropna=True
        )
        return self

    def query_latest_data(self, totalobs, override=False):
        """Query the db and get live data. Work in Progress."""
        print("Querying the latest data...")
        if not override:
            query = """
                select record_date, description_id, description, fulltext_description, port_us, port_origin, shipper, consignee
                from cbp.imports_combined
                order by record_date desc
                limit {}
                ;
            """.format(totalobs)
            self.cur.execute(query)
        # TODO: not sure the best way to implement this
        else:
            query = str(override)
            print(query)
            self.cur.close()
            self.con.close()

        colnames = ["date", "desc_id", "desc", "desc_ft", "port_us", "port_og", "ship", "cons"]
        self.og = pd.DataFrame.from_records(self.cur.fetchall(), columns=colnames)
        self.cur.close()
        self.con.close()
        return self

    def build_sampling_table(self, sample_pct=0.10):
        """Sample some of the latest import data from our db!

        Args:
            sample_pct (float) = percent of latest obs to sample for each hs4 code
        """
        # gets a np.array (str) with all identified hs4 codes & corresponding counts
        query = """
            select hs, count(hs) as hs_count
            from dev.zad_hs_results
            group by hs
            ;
        """
        self.cur.execute(query)
        hs4_code_counts = self.cur.fetchall()
        # create temporary dataframe table in Postgres
        query = """
            drop table if exists dev.zad_construct_sample;
                create table dev.zad_construct_sample as (
                    select *
                    from dev.zad_construct_full
                    limit 0
                )
            ;
        """
        self.cur.execute(query)
        # loop through each hs4 code & sample a percentage
        # of its latest total observations from dev.zad_construct_full.
        print("Sampling from the db...")
        sql_payload = list()
        for hs4_code_count in hs4_code_counts:
            total_sample_obs = math.ceil(sample_pct * int(hs4_code_count[1]))
            # query = """
            #     insert into dev.zad_construct_sample
            #         select *
            #         from dev.zad_construct_full
            #         where hs = \'{hs4_code}\'
            #         limit {sample_obs}
            #     ;
            # """.format(hs4_code=hs4_code_count[0], sample_obs=total_sample_obs)
            query = "insert into dev.zad_construct_sample select * from dev.zad_construct_full " + \
                    "where hs = \'{hs4}\' limit {sample_obs};".format(
                        hs4=hs4_code_count[0], sample_obs=total_sample_obs
                    )
            sql_payload.append(query)
        sql_payload = " ".join(sql_payload)
        sql_payload += " select * from dev.zad_construct_sample;"
        print("Executing sql payload...")
        self.cur.execute(sql_payload)
        self.con.commit()
        # get the results back as original data attribute
        colnames = ["date", "desc_id", "desc", "desc_ft", "port_us", "port_og", "ship", "cons", "hs4"]
        self.og = pd.DataFrame.from_records(self.cur.fetchall(), columns=colnames)
        print("Data constructed from stratified sample.")
        self.cur.close()
        self.con.close()
        return self

    def sample_from_db(self):
        print("Sampling from the db...")
        query = "select record_date::text, " \
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
        self.cur.execute(query)
        # get the results back as original data attribute
        print("{}> Creating data records".format(" "*2))
        start_time = process_time()

        data_records = list()
        batch_size = 100_000
        data_records.extend(self.cur.fetchmany(batch_size))
        while data_records:
            batch = self.cur.fetchmany(batch_size)
            if batch:
                data_records.extend(batch)
            else:
                break

        print("{}> Fetch time:".format(" "*2), process_time() - start_time)
        colnames = ["date", "desc_id", "desc", "desc_ft", "port_us", "port_og", "ship", "cons", "hs4"]
        self.og = pd.DataFrame.from_records(data_records, columns=colnames)
        print(self.og.head(10))
        pdb.set_trace()
        print("Successful pull from sampling table.")
        return self

    def clean(self):
        """Auto clean method that performs all common wrangling methods.

        Returns modified self.df attribute equal to a clean dataframe for experimentation.
        """
        self.df = Wrangler.clean_raw_text(self.og)
        return self

    def stratified_train_test_split(self, train_pct=0.70):
        """Creates training and testing subsets based on a stratified random sample.

        Args:
            train_pct (float): percent of sampled obs to use for training
        """
        self.compute_totals(pandas_df=self.df, partition_on="hs4")

        training_indices = list()
        for hs4 in set(self.df["hs4"]):
            total_training_obs = self.counts[hs4] * train_pct
            subset_indices = list(
                self.df.query("hs4 == " + hs4).sample(n=total_training_obs, random_state=None).index
            )
            training_indices.append(subset_indices)
        training_indices.sort(key=None, reverse=False)

        self.train = self.df.iloc[training_indices]
        self.test = self.df[~self.df.isin(self.train)].dropna(how="all")
        return self


class Wrangler(object):
    """Makes the data work."""

    @staticmethod
    def __concat_text(arglist):
        """Combines text data args provided by arglist."""
        text = str()
        for arg in arglist:
            text += arg
        return text

    @staticmethod
    def __crack_hs_code(pandas_df):
        """Creates two additional columns by breaking down the HS4 code."""
        df = pandas_df
        df["hs2"] = df["hs4"][:2]
        df["hs1"] = df["hs4"][0]
        return df

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
    def clean_raw_text(pandas_df):
        """Performs all standard cleaning methods."""
        df = pandas_df.astype(str)

        # we can start by dropping duplicates records
        df.drop_duplicates(subset='desc_id', keep=False, inplace=True)

        # we need to first standardize the port cols & the shipper/consignee
        # cols so the algorithm won't confuse them with other features.
        # for example, a port_us record with value "New York" becomes: "port_us_NewYork"
        cols = ["port_us", "port_og", "ship", "cons"]
        df_cols = list()
        for col in cols:
            df[col] = Wrangler.__concat_text([col + "_", df[col].str.replace(" ", "")])
            df_cols.append(df[col])

        # format the fulltext description
        df["desc_ft"] = df["desc_ft"].apply(Wrangler.__format_fulltext)

        # now we need to combine all text cols into the one true description plus (TM)
        df_cols = [df[col] for col in df_cols]
        df_cols.insert(0, df["desc_ft"])
        df["descft+"] = Wrangler.__concat_text(df_cols)
        df_cols.pop(0)
        df_cols.insert(0, df["desc"])
        df["desc+"] = Wrangler.__concat_text(df_cols)
        df = Wrangler.__crack_hs_code(df)

        return df[["desc_id", "desc+", "descft+", "hs4", "hs2", "hs1"]]

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
            subset = df.query(hs4_colname + " == " + hs4)[desc_colname]
            subset = subset.str.replace(hs4_pattern, "")
            if sdw_only:
                sdw_pattern = re.compile(r"[^\s\d\w]")
                subset = subset.str.replace(sdw_pattern, " ")
            desc_nohs = desc_nohs.append(subset)
        return desc_nohs.sort_index().str.lower()

    @staticmethod
    def unpack_results(results, base_colname):
        """Unpack those pesky result lists!

        Args:
            results (pd.Series): size n x 1 where each record must be list with length 3
            base_colname (str): column name used as a prefix for the unpacked columns
        Returns: pd.DataFrame of size n x 3 with unpacked result vectors for rows
        """
        buffer = results
        buffer[[base_colname + "_3", base_colname + "_2", base_colname + "_1"]] = pd.DataFrame(
            results.to_list(), index=results.index
        )
        return buffer

    @staticmethod
    def vectorize(xtrain, xtest, stopwords=None, ngrams=(2, 2), maxdf=0.50):
        """Vectorize your training & testing data in one step with this handy method.

        Turns nx1 rows of text data into nxf array of features where f is the
        total number of features identified in the text data by the vectorizer.

        Returned tfidf arrays have tfidf vectors as records which
        can be used as input to our text classification algorithm.
        """
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
        self.results = None

    def train_node(self, xtrain, ytrain, xtest, ytest, nodename, stopwords=None, ngrams=(2, 2), maxdf=0.50):
        # vectorize the training & testing xdata
        tfidf_vec, train_tfidf, test_tfidf = Wrangler.vectorize(xtrain, xtest, stopwords, ngrams, maxdf)
        clf = ComplementNB(alpha=0.05, norm=False)
        clf.fit(train_tfidf, ytrain)

        hspreds, hspreds_c = clf.predict(test_tfidf, return_n=3, return_all=True)
        acc = accuracy_score(ytest, hspreds[:, -1])
        print("{} Node classification accuracy: {}".format(nodename, acc))

        hspreds = pd.Series(hspreds, index=ytest.index)
        hspreds_c = pd.Series(hspreds_c, index=ytest.index)

        self.graph["node_" + nodename] = {"vec": tfidf_vec, "clf": clf}
        self.results["node_" + nodename] = {"acc": acc, "results": hspreds, "results_c": hspreds_c}
        return self.results["node_" + nodename]

    def train_layer(self, train_data, test_data, x_colname, y_colname):
        # initialize data structures to hold results
        layer_results = pd.Series()
        layer_results_c = pd.Series()
        # loop through each y in the input set & train & node for each
        for y in set(train_data[y_colname]):
            query = y_colname + " == " + y

            xtrain = train_data.query(query)[x_colname]
            ytrain = train_data.query(query)[y_colname]

            xtest = test_data.query(query)[x_colname]
            ytest = test_data.query(query)[y_colname]

            node_results = self.train_node(xtrain, ytrain, xtest, ytest, nodename=y)
            layer_results = layer_results.append(node_results[y]["hspreds"])
            layer_results_c = layer_results_c.append(node_results[y]["hspreds_c"])
        # sort results by index
        layer_results.sort_index(inplace=True)
        layer_results_c.sort_index(inplace=True)
        # unpack results list into 3 cols, keeping track of the indices
        # our layer results will become n x 3 pd.DataFrames
        layer_results = Wrangler.unpack_results(layer_results, y_colname)
        layer_results_c = Wrangler.unpack_results(layer_results_c, y_colname)

        acc = accuracy_score(test_data[y_colname], layer_results.iloc[:, -1])
        print("{} Layer classification accuracy: {}".format(y_colname, acc))
        # access specific results with relevant prefix, inner dict with keys: 'acc' | 'results' | 'results_c'
        # example layer level: BayesNet.results['layer_hs2']['acc']
        # example node level: BayesNet.results['node_39']['results']
        self.results["layer_" + y_colname] = {"acc": acc, "results": layer_results, "results_c": layer_results_c}
        return self.results["layer_" + y_colname]

    def train_graph(self, train_data, test_data, input_type="desc+", root_y="hs1"):
        # train root node (predicts hs1)
        xtrain = train_data[input_type]
        ytrain = train_data[root_y]
        xtest = test_data[input_type]
        ytest = test_data[root_y]
        self.train_node(xtrain, ytrain, xtest, ytest, nodename="root")

        # train layer 1 (predicts hs2)
        self.train_layer(train_data, test_data, input_type, "hs2")

        # train layer 2 (predicts hs4)
        self.train_layer(train_data, test_data, input_type, "hs4")

        return self

    def execute_root(self, xdf):
        """Basic method that executes the graph's root node.

        Args:
            xdf (pd.DataFrame): contains 'xdata' col with raw xdata
        Returns: tuple(np.array[class3, class2, class1], np.array[cvalue3, cvalue3, cvalue1])
        """
        x_tfidf = self.graph["node_root"]["vec"].transform(xdf["xdata"])
        root_results, root_results_c = self.graph["node_root"]["clf"].predict(x_tfidf)
        return root_results, root_results_c

    def execute_layer(self, xdf, input_colname):
        layer_results = pd.Series()
        layer_results_c = pd.Series()
        for nodename in set(xdf[input_colname]):
            query = input_colname + " == " + nodename
            subset = xdf.query(query)["xdata"]
            node = self.graph["node_" + nodename]
            x_tfidf = node["vec"].transform(subset)
            node_results, node_results_c = node["clf"].predict(x_tfidf)
            layer_results.append(node_results)
            layer_results_c.append(node_results_c)

        layer_results.sort_index(inplace=True)
        layer_results_c.sort_index(inplace=True)
        return layer_results, layer_results_c

    def execute_graph(self, xdata):
        # format data to receive results
        xdf = pd.DataFrame(xdata, columns=["xdata"])
        del xdata

        # EXECUTE ROOT NODE
        root_results, root_results_c = self.execute_root(xdf)
        # unpack results into dataframes
        root_results = Wrangler.unpack_results(root_results, "hs1")
        root_results_c = Wrangler.unpack_results(root_results_c, "hs1_c")
        # join dataframes
        xdf = xdf.join([root_results, root_results_c], how="left")

        # EXECUTE LAYER 1
        layer_results1, layer_results1_c = self.execute_layer(xdf, input_colname="hs1_1")
        layer_results1 = Wrangler.unpack_results(layer_results1, "hs2")
        layer_results1_c = Wrangler.unpack_results(layer_results1_c, "hs2_c")
        xdf = xdf.join([layer_results1, layer_results1_c], how="left")

        # EXECUTE LAYER 2
        layer_results2, layer_results2_c = self.execute_layer(xdf, input_colname="hs2_1")
        layer_results2 = Wrangler.unpack_results(layer_results2, "hs4")
        layer_results2_c = Wrangler.unpack_results(layer_results2_c, "hs4_c")
        xdf = xdf.join([layer_results2, layer_results2_c], how="left")

        return xdf


def main():
    # activate data object
    server_cursor_name = "adam_cnb_dev"
    print("Server cursor name:", server_cursor_name)
    data = LiveData.connect(server_cursor_name)

    # data.build_sampling_table(sample_pct=0.25)
    data.sample_from_db()

    data.clean()
    data.stratified_train_test_split(train_pct=0.50)
    model = BayesNet()
    model.train_graph(data.train, data.test, input_type="desc+", root_y="hs1")

    # predict for live data
    # data.query_latest_data(totalobs=1000)
    # model.execute_graph(data.df)

    return None


if __name__ == "__main__":
    main()
