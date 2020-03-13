# built-in
from time import process_time
# third-party
import pandas as pd
from joblib import load
# local
import conndb
from cnb_dev import MasterModel
from cnb_dev import MasterDataFrame


##
def load_model(model_path=None) -> MasterModel:
    """Loads in an exported classification model and returns a MasterModel object.

    Args:
        model_path (str): path to an exported .joblib file that persists from cnb_dev
    """
    if not model_path:
        print("[USER INPUT] Please provide the full path to a joblib file containing the model.")
        model_path = input()
    print(">Loading model from specified directory...")
    start_time = process_time()
    model = MasterModel()
    model.models = load(model_path)
    print(">Hash table loaded as: model.models")
    print("--- Loading time: " + str((process_time() - start_time)) + " seconds ---")

    return model


def sql_format(sql_str) -> str:
    """Formats SQL from user input to be easier on the eyes."""
    sql = sql_str.split()
    keywords = ['select', 'from', 'where', 'and', 'or']
    for i in range(len(sql)):
        if sql[i] in keywords:
            sql[i] = '\n' + sql[i]

    return " ".join(sql)


def query_data_to_classify(database_cursor) -> MasterDataFrame:
    """Queries dev db and returns a MasterDataFrame object holding data for classification.

    Args:
        database_cursor: expects a psycopg2 cursor object
    """
    cur = database_cursor
    query = """
        drop table if exists t_df;
        create temp table t_df(
            description_id varchar,  
            description varchar, 
            port_origin varchar, 
            port_us varchar,
            shipper varchar,
            consignee varchar
            )
        """
    cur.execute(query)
    print("Please write a SQL query to select data from the database for classification.")
    print(">The columns you select must adhere to the following order and format:")
    print("\tdescription_id: varchar\n\tdescription: varchar\n\tport_origin: varchar\n\tport_us: varchar\n\t"
          "shipper: varchar\n\tconsginee: varchar")
    
    gate_query = True
    while gate_query:
        print("[USER INPUT] Please submit your query.")
        print(">If you are using puTTY, you may enter 'Shift+Insert' to paste query from clipboard.")
        query = ("insert into t_df {}".format(input()))
        print(">Does this look correct?\n", sql_format(query))
        print("[USER INPUT] Y/N")
        if input() == 'Y':
            gate_query = False
    
    print(">Thank you.")
    print(">Executing query...")
    cur.execute(query)
    print(">Moving the data into Python...")
    query = "select * from t_df"
    cur.execute(query)
    print(">Cleaning up the data...")
    colnames = ['desc_id', 'desc', 'port_orig', 'port_us', 'shipper', 'consignee']
    df = MasterDataFrame(pd.DataFrame.from_records(cur.fetchall(), columns=colnames))
    df.add_desc_cat()
    print(">Query result stored in instance of MasterDataFrame: df.master\n")
    
    return df


def gate_master_internal_redirect(df, model, x_tfidf, redirect_options):
    """The internal control structure for gate_master.

    Args:
        df (MasterDataFrame): holds our data
        model (MasterModel): holds the models and stuff
        x_tfidf (scipy.sparse): x_tfidf for level 0 classification
        redirect_options (list[str]): debug a container desc, submit another query,
            or exit the script entirely.

    Returns:
        df, model, gate_master(str; the chosen redirect option)
    """
    print("[USER INPUT] You will eventually be looped back here unless you enter 'exit'.")
    print(">To export the input data along with their predicted HS codes, enter: " + redirect_options[2])
    print(">To view/export the feature probabilities for a specific description, enter its df index value.")
    print(">To submit another query for classification, enter: " + redirect_options[1])
    print(">To exit this script, enter: " + redirect_options[0])

    gate_master = input()
    if gate_master == redirect_options[0]:
        print('Thank you for using predict_hs.py! :)\n')
    elif gate_master == redirect_options[1]:
        print("Back to the top...\n")
        model.results = None
        df = None
    elif gate_master == redirect_options[2]:
        print("[USER INPUT] Please provide a full export path including file extension (.csv):")
        export_path = input()
        print(">Exporting data...")
        df.master.to_csv(export_path)
        print(">You data can be found at: " + export_path)
    else:
        row_id = int(gate_master)
        # pandas version of x_tfidf (column names are keys for tfidf_vocab) for easy indexing/slicing
        x_tfidf = pd.DataFrame(x_tfidf.toarray())
        # dictionary with keys mapped to word tokens (vocabulary)
        debug_row_metapred = df.master.iloc[row_id, 'metapred']
        tfidf_vocab = {y: x for x, y in model.models[debug_row_metapred][1].vocabulary_.items()}
        # clf_feat_logprob has log probabilities for all features for all classes
        # clf.classes_ for class mapping
        # clf_feat_logprob = pd.DataFrame(clf.feature_log_prob_)
        clf_feat_logprob = pd.DataFrame(model.models[debug_row_metapred][0].feature_log_prob_)
        row_feat = x_tfidf.iloc[row_id:row_id + 1, :].squeeze()
        row_feat = row_feat[row_feat != 0]
        row_feat_probs = clf_feat_logprob[row_feat.index]

        row_vocab = []
        for x in row_feat.index:
            vocab_word = tfidf_vocab[x]
            row_vocab.append(vocab_word)
        row_feat_probs.columns = row_vocab
        row_feat_probs.insert(0, "CLASS", [x for x in model.models[debug_row_metapred][0].classes_], True)
        row_feat_probs.insert(1, "PROB_SUM", row_feat_probs.sum(axis=1), True)
        print(row_feat_probs)

        print("[USER INPUT] Would you like to export row_feat_probs (above) for the current description?: Y/N")
        if input() == 'Y':
            print("[USER INPUT] Please provide a full export path including file extension (.csv):")
            export_path = input()
            # catch incorrect file ext, fails when path includes directory containing '.'
            if export_path.endswith('.csv') is not True:
                export_path = export_path.replace('.', '') + '.csv'
            print(">Exporting data...")
            row_feat_probs.to_csv(export_path)
            print(">Your data can be found at: " + export_path)

    return df, model, x_tfidf, gate_master


def main():
    # connect to dev db
    cur = conndb.connect_db()
    # load in the model
    model = load_model()
    # choosing data to classify
    gate_master = 'open'
    while gate_master != 'exit':
        gate_master = 'open'
        # get the data
        df = query_data_to_classify(database_cursor=cur)
        # predict stuff
        # live_prediction() method returns the x_tfidf used in level0 classification
        x_tfidf = model.live_prediction(df.master)
        print("Preview of results: \n")
        print(df.master[['desc_id', 'desc_cat', 'hs4pred']])
        print("\n")
        # internal redirection loop
        redirect_options = ['exit', 'classify', 'export']
        while gate_master not in redirect_options[:2]:
            df, model, x_tfidf, gate_master = gate_master_internal_redirect(df, model, x_tfidf, redirect_options)


if __name__ == "__main__":
    main()
