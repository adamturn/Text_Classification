# adam note: '##' used for testing in pycharm
##
# built-in
from time import process_time
# local
from cnb_dev import MasterDataFrame
from cnb_dev import MasterModel
# external
import psycopg2
import pandas as pd
from joblib import load

print("Establishing connection with database...")
conn = psycopg2.connect(
    host="xxx-xx-xxx-xxx-xx.compute-1.amazonaws.com",
    database="db",
    port=5432,
    user="usr",
    password="pw"
)
cur = conn.cursor()
print(">Connection established.")

##
print("[USER INPUT] Please provide the full path to a joblib file containing the model.")
model_path = input()
# loading in the MasterModel.models dictionary. Documentation should be available with help().
print(">Loading model from specified directory...")
start_time = process_time()
model = MasterModel()
model.models = load(model_path)
print(">Hash table loaded as: model.models")
print("--- Loading time: " + str((process_time() - start_time)) + " seconds ---")

##
# choosing data to classify
MASTER_GATE = 'open'
while MASTER_GATE != 'exit':
    MASTER_GATE = 'open'
    # creating a temporary table: t_df
    query = '''
        drop table if exists t_df;
        create temp table t_df(
            description_id varchar,
            description varchar,
            port_origin varchar,
            port_us varchar,
            shipper varchar,
            consignee varchar
            )
        '''
    cur.execute(query)
    print("Please write a SQL query to select data from the database for classification.")
    print(">The columns you select must adhere to the following order and format:")
    print("\tdescription_id: varchar\n\tdescription: varchar\n\tport_origin: varchar\n\tport_us: varchar\n\t" +
          "shipper: varchar\n\tconsginee: varchar")

    QUERY_GATE = True
    while QUERY_GATE:
        print("[USER INPUT] Please submit your query.")
        print(">If you are using puTTY, you may enter 'Shift+Insert' to paste query from clipboard.")
        query = ("insert into t_df {}".format(input()))

        def sql_format(some_query):
            sql = some_query.split()
            keywords = ['select', 'from', 'where', 'and', 'or']
            for i in range(len(sql)):
                if sql[i] in keywords:
                    sql[i] = '\n' + sql[i]
            return print(" ".join(sql))

        print(">Does this look correct?\n", sql_format(query))
        print("[USER INPUT] Y/N")
        if input() == 'Y':
            QUERY_GATE = False
    print(">Thank you.")
    print(">Executing query...")
    cur.execute(query)
    print(">Moving the data into Python...")
    query = 'select * from t_df'
    cur.execute(query)
    print(">Cleaning up the data...")
    # adding column names
    colnames = ['desc_id', 'desc', 'port_orig', 'port_us', 'shipper', 'consignee']
    df = MasterDataFrame(pd.DataFrame.from_records(cur.fetchall(), columns=colnames))
    df.add_desc_cat()
    print(">Query result stored in instance of MasterDataFrame: df.master\n")

    # PREDICTION
    X_TFIDF = model.live_prediction(df)  # method operates in the background, returns x_tfidf_level0
    print("Preview of results: \n")
    print(df.master[['desc_id', 'desc_plus', 'hs4pred']])
    print("\n")

    REDIRECT_OPTIONS = ['exit', 'classify', 'export']
    while MASTER_GATE not in REDIRECT_OPTIONS[:2]:
        print("[USER INPUT] You will eventually be looped back here unless you enter 'exit'.")
        print(">To export the input data along with their predicted HS codes, enter: " + REDIRECT_OPTIONS[2])
        print(">To view/export the feature probabilities for a specific description, enter its df index value.")
        print(">To submit another query for classification, enter: " + REDIRECT_OPTIONS[1])
        print(">To exit this script, enter: " + REDIRECT_OPTIONS[0])

        MASTER_GATE = input()
        if MASTER_GATE == REDIRECT_OPTIONS[0]:
            print('Thank you for using predict_hs.py! :)\n')
        elif MASTER_GATE == REDIRECT_OPTIONS[1]:
            print("Back to the top...\n")
            model.results = None
            df = None
        elif MASTER_GATE == REDIRECT_OPTIONS[2]:
            print("[USER INPUT] Please provide a full export path including file extension (.csv):")
            export_path = input()
            print(">Exporting data...")
            df.master.to_csv(export_path)
            print(">You data can be found at: " + export_path)
        else:
            row_id = int(MASTER_GATE)
            # pandas version of x_tfidf (column names are keys for tfidf_vocab) for easy indexing/slicing
            X_TFIDF = pd.DataFrame(X_TFIDF.toarray())
            # dictionary with keys mapped to word tokens (vocabulary)
            debug_row_metapred = df.master.iloc[row_id, 'metapred']
            tfidf_vocab = {y: x for x, y in model.models[debug_row_metapred][1].vocabulary_.items()}
            # clf_feat_logprob has log probabilities for all features for all classes (clf.classes_ for class mapping)
            # clf_feat_logprob = pd.DataFrame(clf.feature_log_prob_)
            clf_feat_logprob = pd.DataFrame(model.models[debug_row_metapred][0].feature_log_prob_)
            row_feat = X_TFIDF.iloc[row_id:row_id + 1, :].squeeze()
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
