import pandas as pd


class Preprocessing:
    def education(self, df):
        level_edu = {
            0: ['Preschool'],  # 0
            1: ['1st-4th', '5th-6th'],  # ~5
            2: ['10th', '9th', '11th'],  # ~7
            3: ['12th', '7th-8th'],  # ~10
            4: [],  # ~15
            5: ['HS-grad'],  # ~17
            6: ['Some-college'],  # ~20
            7: [''],  # ~25
            8: ['Assoc-voc', 'Assoc-acdm'],  # ~30
            9: [''],  # ~35
            10: [''],  # ~40
            11: ['Bachelors'],  # ~45
            12: [''],  # ~50
            13: ['Masters'],  # ~55
            14: [''],  # ~60
            15: [''],  # ~65
            16: [''],  # ~70
            17: ['Doctorate', 'Prof-school'],  # ~75
        }

        def filter_education(X):
            for key, value in level_edu.items():
                if X in value:
                    return key

        df['education'] = df['education'].apply(filter_education)
        return df

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
        # df['native-country'] = df['native-country'].apply(
        #     lambda x: x if x == ' United-States' else 'others')
        level_dic = {
            1: ['Guatemala', 'Columbia', 'Nicaragua', 'Vietnam', 'Trinadad&Tobago', 'Holand-Netherlands'],
            2: ['Honduras', 'Haiti', 'Laos', 'Outlying-US(Guam-USVI-etc)', 'Mexico', 'El-Salvador', 'Dominican-Republic'],
            3: ['Thailand', 'Jamaica', 'Peru', 'Scotland', 'Puerto-Rico'],
            4: ['Ecuador', 'South', 'Portugal', 'Poland'],
            5: ['?', 'Cuba', 'Hong'],
            6: ['Philippines', 'China', 'United-States'],
            7: ['Germany', 'Cambodia', 'Yugoslavia', 'Taiwan', 'France'],
            8: ['Japan', 'England', 'Canada', 'Iran', 'Italy', 'Greece', 'Ireland'],
            9: ['India', 'Hungary'],
            10: []
        }

        def filter_country(X):
            for key, value in level_dic.items():
                if X in value:
                    return key

        df['native-country'] = df['native-country'].apply(filter_country)
        return df

    def capital(self, df):
        df['capital'] = df['capital-gain'] - df['capital-loss']
        df = df.drop(columns=['capital-gain', 'capital-loss'])
        return df

    def race(self, df):
        df['race'] = df['race'].apply(
            lambda x: 1 if x in ['White', 'Asian-Pac-Islander'] else 0)
        return df

    def marital_status(self, df):
        cate = {
            "Married-civ-spouse": 7,
            "Married-AF-spouse": 6,
            "Divorced": 5,
            "Married-spouse-absent": 4,
            "Widowed": 3,
            "Separated": 2,
            "Never-married": 1
        }

        df['marital-status'] = df['marital-status'].map(cate)
        return df

    def oneHot_encoding(self, df):
        df = pd.get_dummies(df, columns=[
            'capital_p',
            'education',
            'workclass',
            'marital-status',
            'occupation',
            'relationship',
            # 'race',
            'sex'
            # 'native-country'
        ])
        return df

    def preprocess_hj(self, df):

        # age : do nothing.

        # workclass
        df = self.workclass(df)

        # fnlwgt
        df = df.drop(columns='fnlwgt')

        # education
        # df = self.education(df)

        # education-num :
        df = df.drop(columns='education-num')

        # marital-status
        df = self.marital_status(df)

        # occupation
        df = self.occupation(df)

        # relationship : do nothing

        # race : do nothing
        df = self.race(df)

        # sex : do nothing
        df['sex'] = df['sex'].apply(lambda x: 1 if x == 'Male' else 0)
        df['maritalXsex'] = df['marital-status'] * df['sex']

        # capital-gain : do nothing
        # capital-loss : do nothing
        df = self.capital(df)
        df['capital_p'] = df['capital'].apply(lambda x: 1 if x == 0 else 2)

        # hours-per-week : do nothing

        # native-country
        df = self.native_country(df)
        df['countryXrace'] = df['race'] * df['native-country']
        # df = df.drop(columns=['capital-gain','capital-loss'])

        # one-hot encoding
        df = self.oneHot_encoding(df)

        return df
