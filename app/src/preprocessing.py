import pandas as pd

target_col = 'binary_target'
all_features = [
    'объем_данных', 'сегмент_arpu', 'on_net',
    'секретный_скор', 'сумма', 'частота',
    'продукт_2', 'продукт_1', 'доход',
    'частота_пополнения', 'pack_freq',
    'pack_or', 'pack_mean'
]

def import_data(path_to_file):
    input_df = pd.read_csv(path_to_file)

    return input_df

def run_preprocessing(input_df):
    train_df = pd.read_csv('./train_data/train.csv')
    threshold = 0.7
    # Удаление столбцов с коэффициентом пропущенных значений выше порога
    train_df = train_df[train_df.columns[train_df.isnull().mean() < threshold]]
    pack_mean_df = pd.DataFrame(dict(train_df.groupby(['pack'])['binary_target'].mean()).items(), columns=['pack', 'pack_mean'])
    ors = {}

    for pack_name in list(train_df['pack'].unique()):
        key = pack_name
        vals = dict(train_df[train_df['pack'] == f'{pack_name}'].binary_target).values()
        ones = len(list(filter(lambda x: x == 1, vals)))
        zeros = len(list(filter(lambda x: x == 0, vals)))
        if ones != 0 and zeros != 0:
            ors[key] = ones / zeros
        elif ones == 0:
            ors[key] = 0
        else:
            ors[key] = 1

    pack_or_df = pd.DataFrame(ors.items(), columns=['pack', 'pack_or'])
    pack_or_df.dropna(inplace=True)
    input_df = input_df.merge(pack_mean_df, how='left', on='pack')
    input_df = input_df.merge(pack_or_df, how='left', on='pack')
    return input_df[all_features]