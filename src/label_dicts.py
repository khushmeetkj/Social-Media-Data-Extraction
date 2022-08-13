import pandas as pd
import joblib
from utils import clean_data
def save_label_dicts(df):
    # remove rows from the DataFrame which do not have corresponding images
    df = clean_data(df)
    # we will use the `gender`, `masterCategory`. and `subCategory` labels
    # mapping `gender` to numerical values
    cat_list_gender = df['gender'].unique()
    # 5 unique categories for gender
    num_list_gender = {cat:i for i, cat in enumerate(cat_list_gender)}
    # mapping `masterCategory` to numerical values
    cat_list_master = df['masterCategory'].unique()
    # 7 unique categories for `masterCategory`
    num_list_master = {cat:i for i, cat in enumerate(cat_list_master)}
    # mapping `subCategory` to numerical values
    cat_list_sub = df['subCategory'].unique()
    # 45 unique categories for `subCategory`
    num_list_sub = {cat:i for i, cat in enumerate(cat_list_sub)}
    # mapping `articleType` to numerical values
    cat_list_article = df['articleType'].unique()
    # 143 unique categories for `articleType`
    num_list_article = {cat:i for i, cat in enumerate(cat_list_article)}
    # mapping `baseColour` to numerical values
    cat_list_base = df['baseColour'].unique()
    # 45 unique categories for `baseColour`
    num_list_base = {cat:i for i, cat in enumerate(cat_list_base)}
    # mapping `season` to numerical values
    cat_list_season = df['season'].unique()
    # 45 unique categories for `prodcutDisplayName`
    num_list_season = {cat:i for i, cat in enumerate(cat_list_season)}
    # mapping `usage` to numerical values
    cat_list_usage = df['usage'].unique()
    # 45 unique categories for `prodcutDisplayName`
    num_list_usage = {cat:i for i, cat in enumerate(cat_list_usage)}
    #######
    joblib.dump(num_list_gender, '../input/num_list_gender.pkl')
    joblib.dump(num_list_master, '../input/num_list_master.pkl')
    joblib.dump(num_list_sub, '../input/num_list_sub.pkl')
    joblib.dump(num_list_article, '../input/num_list_article.pkl')
    joblib.dump(num_list_base, '../input/num_list_base.pkl')
    joblib.dump(num_list_season, '../input/num_list_season.pkl')
    joblib.dump(num_list_usage, '../input/num_list_usage.pkl')
df = pd.read_csv('../input/fashion-product-images-small/style2.csv',usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
df = df.dropna(how='all')
save_label_dicts(df)