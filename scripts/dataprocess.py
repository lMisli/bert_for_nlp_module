
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import os
import pandas as pd


def load_data(data_path, sep=',', header=0):
    ''' Load data from data_path, return DataFrame
    :param
        data_path: data path read from
    :return: DataFrame
    '''
    data = pd.read_csv(filepath_or_buffer=data_path, sep=sep,header= header)
    return data


def split_data_by_rows(input_path, output_path1, output_path2, split_rate=0.7, random=True, random_seed=200):
    data = load_data(input_path)
    if random:
        data1 = data.sample(frac=split_rate, random_state=random_seed)
    else:
        data_len = data.shape[0]
        p1 = int(data_len*split_rate)
        data1 = data.iloc[:p1,:]
    data2 = data.drop(data1.index)
    data1.to_csv(output_path1)
    data2.to_csv(output_path2)





def get_part_data(df:pd.DataFrame, save_name=""):
    part_data = df.iloc[0:5,]
    part_data.to_csv(save_name, index=0)



def clean_text(dataTable, cleanColumns, rmStop, rmPunc, useLemma, save_name=""):
    ''' Clean up the text of the selected cleanColumns in dataTable
    :param dataTable: DataFrame, data contained need to be cleaned.
    :param cleanColums: list, contains columns that text in them need to  be cleaned.
    :param rmStop: boolean, remove the stop words or not, default True.
    :param rmPunc: boolean, remove the punctions or not, default True.
    :param useLemma: boolean, lemmatize words or not, default True.
    :param save_name: str, cleaned dataTable save path, default None, the cleaned dataTable will not be saved
    :return: DataFrame, dataTable after text cleaning
    '''

    def clean(text, rmStop, rmPunc, useLemma):
        #text = text.lower().split()
        if rmStop:
            stopw = set(stopwords.words('english'))
            text = " ". join([i for i in text.lower().split() if i not in stopw])
        if rmPunc:
            exclude = set(string.punctuation)
            text = "".join(ch for ch in text if ch not in exclude)
        if useLemma:
            lemma = WordNetLemmatizer()
            text = " ".join(lemma.lemmatize(word) for word in text.split())
        return text

    for col in cleanColumns:
        #dataTable[col] = dataTable.apply(lambda x: clean(x,rmStop, rmPunc, useLemma))
        dataTable[col] = dataTable[col].map(lambda x: clean(x, rmStop, rmPunc, useLemma))
    if save_name != "":
        dataTable.to_csv(os.path.abspath('.') + '/' + save_name, index=0)
    return dataTable


def combine_test_data(test_data_path:str, test_label_path:str, output_path:str):
    data = load_data(test_data_path)
    label = load_data(test_label_path)
    comment = data[["comment_text"]]
    label.insert(1,column='comment_text', value=comment.values)
    # test_data = pd.concat([label, comment], axis = 1)
    # test_data = test_data.reindex(['id','comment_text', 'toxic','severe_toxic','obscene','threat','insult','identity_hate'])
    label.to_csv(output_path, index=0)


def clean_test_label(test_label_path:str, test_out_path:str):
    data = load_data(test_label_path)
    print(data.shape[0])
    data = data[data.toxic != -1]
    print(data.shape[0])
    data.to_csv(test_out_path)


def balance_data(train_path:str, out_path:str):
    data = load_data(train_path)
    neg_data = data[data.toxic == 0]
    pos_data = data[data.toxic == 1]
    neg_nums = neg_data.shape[0]
    pos_nums = pos_data.shape[0]
    print("negative data:", neg_data.shape[0])
    print("positive data:", pos_data.shape[0])
    if neg_nums > pos_nums:
        sample_rate = pos_nums / neg_nums
        neg_data = neg_data.sample(frac=sample_rate, axis=0)
        print("negative data after sampling:", neg_data.shape[0])
    else:
        sample_rate = neg_nums / pos_nums
        pos_data = pos_data.sample(frac=sample_rate, axis=0)
        print("positive data after sampling:", pos_data.shape[0])

    balanced_data = pd.concat([pos_data, neg_data],axis=0)
    print(balanced_data.head(10))
    balanced_data = balanced_data.sample(frac=1, axis=0)
    print(balanced_data.head(10))
    balanced_data.to_csv(out_path,index=0)





if __name__ == "__main__":
    train_path= "D:\\corpus\\toxic comment\\jigsaw-toxic-comment-classification-challenge\\cleaned_train.csv"
    out_path = "D:\\corpus\\toxic comment\\jigsaw-toxic-comment-classification-challenge\\cleaned_train_sampled.csv"
    balance_data(train_path, out_path)
    # test_label_path = "D:\\corpus\\toxic comment\\jigsaw-toxic-comment-classification-challenge\\test_data_labels.csv"
    # test_out_path = "D:\\corpus\\toxic comment\\jigsaw-toxic-comment-classification-challenge\\cleaned_test_data_labels.csv"
    # clean_test_label(test_label_path, test_out_path)
    # test_label_path = "D:\\corpus\\toxic comment\\jigsaw-toxic-comment-classification-challenge\\test_labels.csv"
    # test_data_path = "D:\\corpus\\toxic comment\\jigsaw-toxic-comment-classification-challenge\\cleaned_test.csv"
    # output_path = "D:\\corpus\\toxic comment\\jigsaw-toxic-comment-classification-challenge\\test_data_labels.csv"
    # combine_test_data(test_data_path, test_label_path, output_path)
    # data = load_data("D:\\corpus\\toxic comment\\jigsaw-toxic-comment-classification-challenge\\cleaned_test_data_labels.csv")
    # #print(data.head(10))
    # #data = clean_text(data,["comment_text"],False, True, True, "cleaned_train.csv")
    # get_part_data(data,"D:\\corpus\\toxic comment\\jigsaw-toxic-comment-classification-challenge\\cleaned_test_labels_5.csv")
    # # # #print(data.head(10))
