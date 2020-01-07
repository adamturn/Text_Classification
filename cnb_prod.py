##
import psycopg2
import numpy as np
import pandas as pd
import math
from time import process_time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score
print("Imports complete.")

print("Establishing connection with database...")
conn = psycopg2.connect(
    host=host,
    database=db,
    port=port,
    user=user,
    password=pw
)
cur = conn.cursor()
print("Connection established.")
##
print("Creating hs_count...")
query = """
    select hs, count(*) as total
    from dev.zad_hs_results
    group by hs
    order by total
"""
cur.execute(query)
# hs_count[:, 0] are hs, hs_count[:, 1] are counts
hs_count = np.array(cur.fetchall())
hs_count = hs_count[:100, :]
print("hs_count created.")

##
# creating master dataframe
# CONSTANTS
sample_pct = 0.50   # percent of total obs sampled for each HS code
train_pct = 0.70    # percent of sampled obs used for training (complement used for testing)

# initializing final data structure, t_df
print("Creating t_df in the database...")

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

# construction loop, sampling latest rows_for_each_hs
print("Constructing data...")

start_time = process_time()

for i in range(hs_count.shape[0]):
    hs_code = hs_count[i, 0]
    total_sample_obs = math.ceil(sample_pct * int(hs_count[i, 1]))
    query = """
    drop table if exists t_construct;
        create temp table t_construct as (
            select a.description_id, a.description, a.port_origin_name, a.port_us_name, a.shipper, a.consignee, b.hs
            from cbp.imports_combined a
            left join dev.zad_hs_results b on a.description_id = b.description_id
            where b.hs = \'""" + hs_code + """\'
            limit """ + str(total_sample_obs) + """
        )
    ;
    insert into t_df
    select * from t_construct
    ;
    """
    cur.execute(query)

query = "select * from t_df"
cur.execute(query)

construct_time = process_time() - start_time

cols = ['desc_id', 'desc', 'port_origin', 'port_us', 'shipper', 'consignee', 'hs']
df = pd.DataFrame.from_records(cur.fetchall(), columns=cols)
print("Dataframe constructed from stratified sample.")

# 'hs2' is a meta category containing 2-digit HTS codes
df['hs2'] = df['hs'].str[:2]
print("New column generated: 'hs2'")


def colcat(dataframe, column_names):
    """
    Concatenates several columns of text data in a pandas Dataframe.
    :param dataframe: pandas Dataframe object
    :param column_names: iterable that contains column names
    :return: new_col: pandas Series object
    """
    dataframe = dataframe.fillna('')
    new_col = dataframe[column_names[0]].values
    for name in column_names[1:]:
        new_col += ' ' + dataframe[name].values
    return new_col


cols_cat = ['desc', 'port_origin', 'port_us', 'shipper', 'consignee']
df['desc_cat'] = colcat(df, cols_cat)
print("New column generated: 'desc_cat'")

hs_count_ptls = np.percentile(hs_count[:, 1].astype(int), [25, 50, 90])
d_hs_vol = {}
for i in range(hs_count.shape[0]):
    if int(hs_count[i, 1]) <= hs_count_ptls[0]:
        d_hs_vol[hs_count[i, 0]] = 0
    elif int(hs_count[i, 1]) <= hs_count_ptls[1]:
        d_hs_vol[hs_count[i, 0]] = 1
    else:
        d_hs_vol[hs_count[i, 0]] = 2


def get_meta_vol(hts_code):
    new_col = []
    for hts in hts_code:
        new_col.append(d_hs_vol[hts])
    return new_col


df['vol'] = get_meta_vol(df['hs'])
print("New column generated: 'vol'")


def strip_hs(hts_code, description):
    pattern = r'[^\d]' + hts_code + r'([^\d]|\d{2}[^\d]|\d{4}[^\d]|\d{6}[^\d])'
    stripped = re.sub(pattern, ' ', description)
    return stripped


df['desc_nohs'] = df.apply(lambda row: strip_hs(row['hs'], row['desc_cat']), axis=1)
print("New column generated: 'desc_nohs'")

print("df created.")

##
# train/test split
# TODO: consider changing this basic train/test split to k-fold cross validation
print("Splitting train/test data...")

df_train = pd.DataFrame()

for i in range(hs_count.shape[0]):
    hs_code = hs_count[i, 0]
    obs_for_each_hs = math.ceil(sample_pct*train_pct * int(hs_count[i, 1]))
    pram = "hs == '" + str(hs_code) + "'"
    df_train = df_train.append(df.query(pram).sample(n=obs_for_each_hs))
    print(df_train[['desc_id', 'desc_nohs', 'hs', 'hs2', 'vol']])

df_test = df[~df.isin(df_train)].dropna(how='all')

# TODO: test 'meta' classes (y data) for volume-based classification at the top level
test_y_vol = df_test['vol']
train_y_vol = df_train['vol']

# 'meta' classes (y data) for 2-digit classification
test_y_hs2 = df_test['hs2']
train_y_hs2 = df_train['hs2']

# x, y data for 4-digit classification train/test
test_y = df_test['hs']
train_y = df_train['hs']

test_x = df_test['desc_nohs']
train_x = df_train['desc_nohs']

print("Training subset created.")
print("Data construction time: ", construct_time)

##
# PART 2: META CLASSIFICATION TREE USING 2 DIGIT HS CODES FROM THE GROUND UP
# TODO: volume based classification test
print("Vectorizing the training data for meta vol classifier...")
tfidf_vec_vol = TfidfVectorizer(
    strip_accents='ascii',
    lowercase=True,
    preprocessor=None,
    tokenizer=None,
    analyzer='word',
    stop_words=None,
    smooth_idf=True,
    sublinear_tf=True,
    ngram_range=(1, 2),
    max_df=0.25,
    min_df=1
)
train_x_vol_tfidf = tfidf_vec_vol.fit_transform(train_x, train_y_vol)

# creating and training the meta vol classifier
print("Creating the meta vol classifier...")
clf_vol = ComplementNB(alpha=1, norm=True)
print("Training the meta vol classifier...")
clf_vol.fit(train_x_vol_tfidf, train_y_vol)

# testing accuracy of meta vol classifier
print("Vectorizing the testing data for the meta vol classifier...")
test_x_vol_tfidf = tfidf_vec_vol.transform(test_x)
print("Predicting volume category with the meta vol classifier...")
clf_vol_pred = clf_vol.predict(test_x_vol_tfidf)
clf_vol_accuracy = accuracy_score(test_y_vol, clf_vol_pred)
print("Meta Classifier Accuracy: ", str(clf_vol_accuracy))

##
# adding volume predictions to master dataframe (0: <= 50th percentile, 1: 50 <= 80, 2: 80 <= 100)
df_test['vol_pred'] = clf_vol_pred

# TODO: going down another level
# STEP 3: creating and training the second tier of classifiers
# creating a dictionary with vol code as the key mapped to associated classifier
print("Initializing clf_models dictionary...")
clf_models = {}
print("Populating clf_models with meta vol models")
for vol in set(df_train['vol']):
    tfidf_vec = TfidfVectorizer(
        strip_accents='ascii',
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        analyzer='word',
        stop_words=None,
        smooth_idf=True,
        sublinear_tf=True,
        ngram_range=(1, 2),
        max_df=0.25,
        min_df=1
    )
    clf = ComplementNB(alpha=1)

    train_x = df_train.query("vol == \'" + str(vol) + "\'")['desc_nohs']
    train_y = df_train.query("vol == \'" + str(vol) + "\'")['hs2']
    train_x_tfidf = tfidf_vec.fit_transform(train_x, train_y)

    model = clf.fit(train_x_tfidf, train_y)
    clf_models[vol] = [model, tfidf_vec]
# adding the meta vol classifier to clf_models
clf_models['vol'] = [clf_vol, tfidf_vec_vol]

# clf_results := {vol: [clf_pred, accuracy]}
clf_results = {}
for vol in set(df_test['vol']):
    test_x = df_test.query("vol == '" + str(vol) + "'")['desc_nohs']
    test_x_tfidf = clf_models[vol][1].transform(test_x)
    test_y = df_test.query("vol == '" + str(vol) + "'")['hs2']

    clf_pred = clf_models[vol][0].predict(test_x_tfidf)
    accuracy = accuracy_score(test_y, clf_pred)
    vol_result = {vol: [clf_pred, accuracy]}
    clf_results.update(vol_result)
    print("Volume code: " + str(vol) + "; Classifier Accuracy: " + str(accuracy))

print("END OF TREE METHOD")
