import os
import pandas as pd

class dataProcess():
    def __init__(self,traindir,testdir):
        self.train = pd.read_csv(traindir)
        self.test = pd.read_csv(testdir)

    def preprocess(self):
        #clean empty rows
        empty_title = ((self.train['title2_zh'].isnull()) | (self.train['title1_zh'].isnull()) | (self.train['title2_zh'] == '') | (self.train['title2_zh'] == '0'))
        self.train = self.train[~empty_title]

        #長度超過50就截掉
        MAX_LENGTH = 50
        self.train = self.train[~(self.train.title1_zh.apply(lambda x : len(x)) > MAX_LENGTH)]
        self.train = self.train[~(self.train.title2_zh.apply(lambda x : len(x)) > MAX_LENGTH)]

        #留下中文去掉英文的
        self.train = self.train.reset_index()
        self.train = self.train.loc[:, ['title1_zh', 'title2_zh', 'label']]
        self.train.columns = ['text_a', 'text_b', 'label']

        self.test = self.test.loc[:, ["title1_zh", "title2_zh", "id"]]
        self.test.columns = ["text_a", "text_b", "Id"]
        # 存成tsv讓PyTorch用
        self.test.to_csv("data/test.tsv", sep="\t", index=False)
        self.train.to_csv("data/train.tsv", sep="\t", index=False)

    def RatioOfLabel(self):
        ratio = self.train.label.value_counts() / len(self.train)
        return ratio
    def RatioOfTRAINTEST(self):
        ratio = len(df_test) / len(df_train)
        return ratio
    def get_data(self):
        return self.train , self.test