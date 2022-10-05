import pandas as pd


class Preprocessing:
    def workclass(self, df):
        wc_level_dic = {
            0: ['Never-worked'],  # 0~5
            1: ['Without-pay'],  # 5~10
            2: ['?'],  # 10~15
            3: [],  # 15~20
            4: ['Private'],  # 20~25
            5: ['State-gov', 'Self-emp-not-inc', 'Local-gov'],  # 25~30
            6: [],  # 30~35
            7: ['Federal-gov'],  # 35~40
            8: [],  # 40~45
            9: [],  # 45~50
            10: [],  # 50~55
            11: ['Self-emp-inc']  # 55~
        }

        def filter_workclass(X):
            for key, value in wc_level_dic.items():
                if str(X).strip() in value:
                    return key
        df['workclass'] = df['workclass'].apply(filter_workclass)
        return df

    def occupation(self, df):
        occupation_level_dic = {
            0: ['Other-service', 'Priv-house-serv'],
            1: ['Handlers-cleaners'],
            2: ['Adm-clerical', 'Machine-op-inspct', 'Farming-fishing', '?'],
            3: [],
            4: ['Craft-repair', 'Transport-moving'],
            5: ['Tech-support', 'Sales'],
            6: ['Protective-serv', 'Armed-Forces'],
            7: [],
            8: ['Prof-specialty'],
            9: ['Exec-managerial']
        }

        def filter_country(X):
            for key, value in occupation_level_dic.items():
                if X in value:
                    return key

        df['occupation'] = df['occupation'].apply(filter_country)
        return df

    def native_country(self, df):
        df['native-country'] = df['native-country'].apply(
            lambda x: x if x == ' United-States' else 'others')
        # level_dic = {
        #     0: ['Guatemala', 'Columbia', 'Nicaragua', 'Vietnam', 'Trinadad&Tobago', 'Holand-Netherlands'],
        #     1: ['Honduras', 'Haiti', 'Laos', 'Outlying-US(Guam-USVI-etc)', 'Mexico', 'El-Salvador', 'Dominican-Republic'],
        #     2: ['Thailand', 'Jamaica', 'Peru', 'Scotland', 'Puerto-Rico'],
        #     3: ['Ecuador', 'South', 'Portugal', 'Poland'],
        #     4: ['?', 'Cuba', 'Hong'],
        #     5: ['Philippines', 'China', 'United-States'],
        #     6: ['Germany', 'Cambodia', 'Yugoslavia', 'Taiwan', 'France'],
        #     7: ['Japan', 'England', 'Canada', 'Iran', 'Italy', 'Greece', 'Ireland'],
        #     8: ['India', 'Hungary'],
        #     9: []
        # }

        # def filter_country(X):
        #     for key, value in level_dic.items():
        #         if X in value:
        #             return key

        # df['native-country'] = df['native-country'].apply(filter_country)
        return df

    # def oneHot_encoding(self, df):
    #     df = pd.get_dummies(df, columns=[
    #         'workclass',
    #         'marital-status',
    #         'occupation',
    #         'relationship',
    #         'race',
    #         'sex',
    #         'native-country'
    #     ])
    #     return df

    def oneHot_encoding(self, df):
        df = pd.get_dummies(df, columns=[
            'workclass',
            'marital-status',
            'occupation',
            'relationship',
            'race',
            'sex',
            'native-country'
        ])
        return df

    def preprocess_hj(self, df):

        # age : do nothing.

        # workclass
        df = self.workclass(df)

        # fnlwgt
        df = df.drop(columns='fnlwgt')

        # education
        df = df.drop(columns='education')

        # education-num : do nothing

        # marital-status : do nothing

        # occupation
        df = self.occupation(df)

        # relationship : do nothing

        # race : do nothing

        # sex : do nothing

        # capital-gain : do nothing

        # capital-loss : do nothing

        # hours-per-week : do nothing

        # native-country
        df = self.native_country(df)

        # one-hot encoding
        df = self.oneHot_encoding(df)

        return df
