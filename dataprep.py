## import
import psycopg2
import pandas as pd

## Dev db connection and creating the cursor
conn = psycopg2.connect(
    host="HOSTNAME",
    database="DATABASE",
    port=5432,
    user="clipper_intern",
    password="PASSWORD"
    )

cur = conn.cursor()

## DATA SET CREATION: 5000 latest records for top 100 HS codes (10/2/2019)
# creating the list of hs codes to sample from
query = '''
    drop table if exists t_hs_list;
    create temp table t_hs_list as (
        select hs, count(*) as total_count
        from cbp.import_description_training_set_customs
        group by hs
        order by total_count desc
        limit 100
    )
    ;
    select * from t_hs_list
    '''
cur.execute(query)
t_hs = cur.fetchall()
t_hs = [x[0] for x in t_hs]

##
# initializing final data structure, t_df
query = '''
    drop table if exists t_df;
    create temp table t_df(description_id varchar, description varchar, hs varchar)
    '''
cur.execute(query)

# construction loop, sampling latest 5000 rows for each hs code
for i in t_hs:
    i = str(i)
    query = '''
        drop table if exists t_construct;
        create temp table t_construct as (
            select description_id, description, hs
            from cbp.import_description_training_set_customs
            where hs = \'''' + i + '''\'
            limit 5000
            )
        ;
        insert into t_df
        select * from t_construct
        ;
        '''
    cur.execute(query)

# fetching the final data table --> t_df
query = 'select * from t_df'
cur.execute(query)
t_df = cur.fetchall()

# pandas for convenience
colnames = ['desc_id', 'desc', 'hs']
t_df = pd.DataFrame.from_records(t_df, columns=colnames)

## export (this handles desc containing ',' by wrapping the entire desc in quotes)
t_df.to_csv("C:\\Users\\Admin\\Documents\\data\\text_classification.csv")

## train(70), test(30) split
t_df_train = pd.DataFrame()
sample_size = 3500

for i in t_hs:
    pram = "hs == \'" + i + "\'"
    t_df_train = t_df_train.append(t_df.query(pram).sample(n=sample_size))
    print(t_df_train)

t_df_test = t_df[~t_df.isin(t_df_train)].dropna(how='all')

train_y = t_df_train['hs']
train_x = t_df_train.drop(columns=['desc_id', 'hs'])

test_y = t_df_test['hs']
test_x = t_df_test.drop(columns=['desc_id', 'hs'])
