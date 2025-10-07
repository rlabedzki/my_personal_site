import pandas as pd

def format_sex(df):
    df.loc[:, "Sex"] = df["Sex"].map({"male": 0, "female": 1}).astype(int)
    return df

def format_name_to_title_age(df):
    df.loc[:, "Title"] = df["Name"].str.extract(r",\s*([^.]*)\.")
    main_titles = ["Mr", "Miss", "Mrs", "Master"]
    df.loc[:, "TitleGroup"] = df["Title"].apply(lambda x: x if x in main_titles else "Other")
    df["Age"] = df.groupby("TitleGroup")["Age"].transform(lambda x: x.fillna(x.median()))
    df = pd.get_dummies(df, columns=["TitleGroup"], drop_first=True)
    return df

def format_embarked(df):
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)
    return df

#for col in features:
#    if col not in df.columns:
#        df[col] = 0  # dodaj brakującą kolumnę z samymi zerami
    
