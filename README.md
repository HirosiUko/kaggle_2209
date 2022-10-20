# kaggle_2209

## 파일 설명

> 문제 solution은 classification220926 폴더 안에 있는 파일 입니다.

**Kaggle_Train_simple.ipynb** : Train을 빨리 시키기 위해서 만든 최종 file

~~kaggle_testset.ipynb~~ : 상기 train으로 부터 만들어진 model을 load하여 test set을 predict파일을 생성하는 file (deprecated)

~~kaggle_train.ipynb~~ : 초기 문제풀기 위해 연구했던 File

~~mylib.py~~ : Data 사전 처리를 단순화 하기 위해 만든 lib임

**preprocessing.py** : 상기 mylib 을 발전시킨 최종 파일, 여기에서 전처리 최종본

**sample_submission.csv** : 제출용 csv를 만들기 위한 templete

**train.csv** : 주워진 Train set

**train.csv** : 주워진 test set


## 테스트 방법

### 1. Kaggle_Train_simple.ipynb 을 수행한다.
```python
from preprocessing import Preprocessing 
pp = Preprocessing()
X_df = pp.preprocess_hj(X_df)
```
상기 preprocessing에서 전처리를 거친다.

### 2. 생성된 파일 선택
```python
make_submission(xgb_clf,"xgb")
```
을 통해하여 `submission_xgb.csv` 이 만들어진다.
xgb model이 가장 accuracy가 높으므로 이것으로 캐글 제출한다.


