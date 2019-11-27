import pandas as pd


def build(out_file):
    data_dir = '../data/'
    labels_a = 'labels-levela.csv'
    labels_b = 'labels-levelb.csv'
    labels_c = 'labels-levelc.csv'
    test_data_a = 'testset-levela.tsv'
    test_data_b = 'testset-levelb.tsv'
    test_data_c = 'testset-levelc.tsv'

    test_a = pd.read_csv(data_dir + test_data_a, sep='\t')
    test_b = pd.read_csv(data_dir + test_data_b, sep='\t')
    test_c = pd.read_csv(data_dir + test_data_c, sep='\t')

    label_a = pd.read_csv(data_dir + labels_a, header=None)
    label_b = pd.read_csv(data_dir + labels_b, header=None)
    label_c = pd.read_csv(data_dir + labels_c, header=None)

    label_a.columns = ['id', 'subtask_a']
    label_b.columns = ['id', 'subtask_b']
    label_c.columns = ['id', 'subtask_c']

    test_data = pd.concat([test_a, test_b, test_c])
    test_data.drop_duplicates(inplace=True)
    print(f'{test_data.shape[0]} unique tweets in test set')

    test = pd.merge(test_data, label_a, on='id')
    test = pd.merge(test, label_b, on='id', how='left')
    test = pd.merge(test, label_c, on='id', how='left')

    test.set_index('id', inplace=True, drop=True)

    test.to_csv(data_dir + out_file, sep='\t', na_rep='NULL')
    print(f'Result saved in {data_dir + out_file}')

if __name__ == "__main__":
    out_file = 'olid-test-v1.0.tsv'
    build(out_file)