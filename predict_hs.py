## import
print("Begin import process...")
import psycopg2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from joblib import load
from time import process_time
print(">Import process complete.")

## connecting to the database. execute queries with cur.execute('query')
print("Establishing connection with database...")

conn = psycopg2.connect(
    host='hostname',
    database='database',
    port=5432,
    user='user',
    password='password'
    )

cur = conn.cursor()
print(">Connection established.")

## defining paths to model and vectorizer
print("[USER INPUT] Please provide the full path to a classification model. (.joblib)")
model_path = input()
#print("[USER INPUT] Please provide the full path to a TF-IDF vectorizer. (.joblib)")
print("This script assumes a paired naming/storage scheme and will automatically load TF-IDF vectorizer.")
tfidf_path = model_path[:-7] + '_vec.joblib'

## loading in the classifier
print(">Loading the classifier from specified directory...")
start_time = process_time()
clf = load(model_path)
print(">Classifier loaded as: clf")
print("--- Loading time: " + str((process_time() - start_time)) + " seconds ---")

## loading in the tf-idf vectorizer
print(">Loading the vectorizer from specified directory...")
start_time = process_time()
tfidf_vec = load(tfidf_path)
print(">TF-IDF Vectorizer loaded as tfidf_vec")
print("--- Loading time: " + str((process_time() - start_time)) + " seconds ---")

## choosing data to classify
master_input = 'start'
while master_input != 'exit':
    master_input = 'start'
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
    print("    description_id: varchar\n    description: varchar\n    port_origin: varchar\n    port_us: varchar\n    shipper: varchar\n    consginee: varchar")

    print("[USER INPUT] Please submit your query.")
    print(">If you are using puTTY, you may enter 'Shift+Insert' to paste query from clipboard.")
    query = ('''
        insert into t_df
        ''' + str(
        input()
        )
    )
    print(">Thank you.")
    print(">Executing query...")
    cur.execute(query)
    print(">Moving the data into Python...")
    query = 'select * from t_df'
    cur.execute(query)
    t_df = cur.fetchall()
    print(">Cleaning up the data...")
    # adding column names
    colnames = ['desc_id', 'desc', 'port_orig', 'port_us', 'shipper', 'consignee']
    df = pd.DataFrame.from_records(t_df, columns=colnames)
    df['desc_plus'] = df[df.columns[1:6]].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
    df.columns = df.columns.astype(str)
    print(">Query result stored as pandas dataframe: df\n")

    print("Creating sparse TF-IDF matrix from data in df.")
    x_tfidf = tfidf_vec.transform(df['desc_plus'])
    print(">Sparse TF-IDF matrix generated successfully: x_tfidf\n")

    ## predicting outcomes
    print("Predicting outcomes for test data...")
    start_time = process_time()
    clf_pred = clf.predict(x_tfidf)
    print(">Prediction complete.")
    print("--- Prediction time: " + str((process_time() - start_time)) + " seconds ---")

    print("Preview of results: \n")
    df['predicted_hs'] = clf_pred
    print(df[['desc_id', 'desc_plus', 'predicted_hs']])
    print("\n")

    redirect_options = ['exit', 'classify', 'export results']
    while master_input not in redirect_options[:2]:
        print("[USER INPUT] You will eventually be looped back here unless you enter 'exit'.")
        print(">To export the input data along with their predicted HS codes, enter: " + redirect_options[2])
        print(">To view/export the feature probabilities for a specific description, enter its index value.")
        print(">To submit another query for classification, enter: " + redirect_options[1])
        print(">To exit this script, enter: " + redirect_options[0])

        master_input = input()
        if master_input == redirect_options[0]:
            print('Thank you for using predict_hs.py! :)')
            print()
        elif master_input == redirect_options[1]:
            print("Back to the top...\n")
            del df
            del clf_pred
        elif master_input == redirect_options[2]:
            print("[USER INPUT] Please provide a full export path including file extension (.csv):")
            export_path = input()
            print(">Exporting data...")
            df.to_csv(export_path)
            print(">You data can be found at: " + export_path)
        else:
            row_id = int(master_input)
            # pandas version of x_tfidf (column names are keys for tfidf_vocab) for easy indexing/slicing
            x_tfidf_pd = pd.DataFrame(x_tfidf.toarray())
            # dictionary with hash keys mapped to word tokens (vocabulary)
            tfidf_vocab = {y: x for x, y in tfidf_vec.vocabulary_.items()}
            # clf_feat_logprob has log probabilities for all features for all classes (clf.classes_ for class mapping)
            clf_feat_logprob = pd.DataFrame(clf.feature_log_prob_)
            row_feat = x_tfidf_pd.iloc[row_id:row_id+1,:].squeeze()
            row_feat = row_feat[row_feat != 0]
            row_feat_probs = clf_feat_logprob[row_feat.index]

            row_vocab = []
            for x in row_feat.index:
                vocab_word = tfidf_vocab[x]
                row_vocab.append(vocab_word)

            row_feat_probs.columns = row_vocab
            row_feat_probs.insert(0, "CLASS", [x for x in clf.classes_], True)
            row_feat_probs.insert(1, "PROB_SUM", row_feat_probs.sum(axis=1), True)
            print(row_feat_probs)

            print("[USER INPUT] Would you like to export row_feat_probs (above) for the current description?: Y/N")
            export_rfp = input()
            if export_rfp == 'Y':
                print("[USER INPUT] Please provide a full export path including file extension (.csv):")
                export_path = input()
                # catch forgotten file ext, might want to add functionality to remove incorrect file ext and replace
                if export_path[-4:] != '.csv':
                    export_path += '.csv'
                print(">Exporting data...")
                row_feat_probs.to_csv(export_path)
                print(">Your data can be found at: " + export_path)
				