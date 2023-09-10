![image](https://github.com/plintAn/Churn_EDA_Solution/assets/124107186/98a2b08d-b9c6-4f80-af35-812fa39c3231)# Churn_EDA_Solution
통신사 고객 이탈에 대한 탐색적 분석 방법을 진행하고 결과를 도출하여 해결 방안을 제시하고 배포하여 이탈 방지 기여도를 작성합니다.

# 데이터 세트 설명

"고객을 유지하기 위한 행동 예측하기. 모든 관련 고객 데이터를 분석하고 집중적인 고객 유지 프로그램을 개발할 수 있습니다." [IBM 샘플 데이터 세트]

데이터 세트 출처(https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)

데이터 목록
* WA_Fn-UseC_-Telco-Customer-Churn.csv

데이터셋은 다음 정보를 포함하고 있습니다:

* 지난달 안에 떠난 고객 - 해당 열은 'Churn'으로 불립니다.
* 각 고객이 가입한 서비스 - 전화, 다중 회선, 인터넷, 온라인 보안, 온라인 백업, 장치 보호, 기술 지원, 스트리밍 TV 및 영화 등
* 고객 계정 정보 - 고객이 된 기간, 계약, 결제 방법, 비종이 청구, 월별 요금, 총 요금 등
* 고객에 대한 인구 통계 정보 - 성별, 연령대, 파트너 및 부양 가족 유무 등


<!-- 목차 -->

# 차 례

| 번호 | 내용                                             |
|------|--------------------------------------------------|
|  I  | [데이터 준비 및 분석](#1)                          |
| II  | [모델링](#2)                                     |
| III | [결과 해석 및 적용](#3)                         |
| IV  | [배포 및 적용](#4)                              |



<!-- intro -->
<div id="1">

# I. 데이터 준비 및 분석

라이브러리 임포트

```python
import pandas as pd
import pandas as np
import matplotlib.pyplot as plt
import seabron as sns
```

밸류 및 타입 확인

```python
df.columns.values
df.dtypes
```

object 열에 대해 숫자형으로 변환 및 결측값 처리

```python
# TotalCharges 컬럼을 숫자로 변환하고 변환 시 에러가 발생하면 NaN으로 처리
df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')

# 각 컬럼별로 NaN 값의 개수를 계산
df.isnull().sum()
```

ID 열은 삭제 후 따로 저장.(추후 시계열 분석시 유용)

```python
df.dropna(inplace = True)
df2 = df.iloc[:,1:] #첫열 삭제(ID)
```

타겟 변수 표준화

```python
df2['Churn'].replace(to_replace='Yes', value=1, inplace=True)
df2['Churn'].replace(to_replace='No',  value=0, inplace=True)
```

### 원-핫 인코딩(더미형 변수 생성)

```python
df_dummies = pd.get_dummies(df2)
df_dummies.head()
```

```python
	SeniorCitizen	tenure	MonthlyCharges	TotalCharges	Churn	gender_Female	gender_Male	Partner_No	Partner_Yes	Dependents_No	...	StreamingMovies_Yes	Contract_Month-to-month	Contract_One year	Contract_Two year	PaperlessBilling_No	PaperlessBilling_Yes	PaymentMethod_Bank transfer (automatic)	PaymentMethod_Credit card (automatic)	PaymentMethod_Electronic check	PaymentMethod_Mailed check
0	0	1	29.85	29.85	0	True	False	False	True	True	...	False	True	False	False	False	True	False	False	True	False
1	0	34	56.95	1889.50	0	False	True	True	False	True	...	False	False	True	False	True	False	False	False	False	True
2	0	2	53.85	108.15	1	False	True	True	False	True	...	False	True	False	False	False	True	False	False	False	True
3	0	45	42.30	1840.75	0	False	True	True	False	True	...	False	False	True	False	True	False	True	False	False	False
4	0	2	70.70	151.65	1	True	False	True	False	True	...	False	True	False	False	False	True	False	False	True	False
5 rows × 46 columns
```

### 기본 시각화

`이탈 여부' 기준 변수 시각화

```python
plt.figure(figsize=(15,8))
df_dummies.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')
```

Output

![image](https://github.com/plintAn/Churn_EDA_Solution/assets/124107186/4094f2d6-3a15-4b12-8809-b9b81608af54)

### 해석

확인 결과 Contract_Month-to-month, OnlineSecurity_No 등 한달 마다 계약, 온라인 서비스 사용 유무에 따라 강한 양의 선형 관계를 확인했고 이는 단기 계약 혹은 온라인 서비스를 사용하지 않는 사람은 '이탈 여부'가 높다는 해석이다. 그리고 오른쪽 사이드인 Tenure, Contract_year 같은 '가입 기간', '1년단위 계약'인 경우 음의 상관관계로 오히려 '이탈 여부'가 줄어드는 결과를 확인했다.

### 성별에 따른 '이탈 여부' 시각화

```python
# 필요한 라이브러리 임포트
import matplotlib.ticker as mtick

# 바 그래프에 사용할 색상 정의
colors = ['#4D3425','#E4512B']

# 'gender' 컬럼의 값 분포를 바 그래프로 표현. 각 값의 비율을 계산하여 표시.
ax = (df['gender'].value_counts()*100.0 /len(df)).plot(kind='bar',stacked = True, rot = 0, color = colors)

# y축을 백분율로 표시
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylabel('% Customers')
ax.set_xlabel('Gender')
ax.set_ylabel('% Customers')
ax.set_title('Gender Distribution')

# 각 막대의 전체 너비를 수집할 리스트 생성
totals = []

# 각 막대의 너비를 찾아 리스트에 추가
for i in ax.patches:
    totals.append(i.get_width())

# 모든 막대의 총 너비 계산
total = sum(totals)

# 각 막대 위에 해당 막대의 백분율 값을 표시
for i in ax.patches:
    # get_width는 좌우로 이동, get_y는 상하로 이동
    ax.text(i.get_x()+.15, i.get_height()-3.5, \
            str(round((i.get_height()/total), 1))+'%',
            fontsize=12,
            color='white',
           weight = 'bold')
```

Output

![image](https://github.com/plintAn/Churn_EDA_Solution/assets/124107186/fb752aac-6428-4afe-b22f-117b552ccfa5)

시각화 결과 성별에 따른 '이탈 여부'는 신경쓰지 않아도 된다고 판단된다.

## 1.가설 설정

기존에 가지고 있던 도메인 지식을 바탕으로 타겟 변수에 영향을 크게 미칠 것 같은 변수를 기준으로 가설을 설정

## 가설:

* 가설1.나이에 따른 이탈여부: 나이가 많은 고객은 이탈할 확률이 적을 것이다.
* 가설2.부양가족 및 파트너 연관 이탈여부: 부양가족이 있는 고객들은 이탈할 확률이 적을 것이다.
* 가설3.사용기간이 긴 고객들은 이탈여부가 적을 것이다.

### 가설1
* 가설1.나이에 따른 이탈여부: 나이가 많은 고객은 이탈할 확률이 적을 것이다.


```python
# 'SeniorCitizen' 컬럼의 값 분포를 파이 차트로 표현. 각 값의 비율을 계산하여 표시.
ax = (df['SeniorCitizen'].value_counts()*100.0 /len(df))\
.plot.pie(autopct='%.1f%%', labels = ['No', 'Yes'],figsize =(5,5), fontsize = 12 ) 

# y축을 백분율로 표시
ax.yaxis.set_major_formatter(mtick.PercentFormatter())

# y축 라벨 설정
ax.set_ylabel('Senior Citizens',fontsize = 12)

# 파이 차트 제목 설정
ax.set_title('% of Senior Citizens', fontsize = 12)

```

Output

![image](https://github.com/plintAn/Churn_EDA_Solution/assets/124107186/2da22887-dfca-45db-997e-bbd887d61db2)

**나이가 많을수록 '이탈 여부'가 적다.**

### 가설2
* 가설2.부양가족 및 파트너 연관 이탈여부: 부양가족이 있는 고객들은 이탈할 확률이 적을 것이다.

```python
# 바 그래프 색상 설정
colors = ['#4D3425','#E4512B']

# 'Partner'와 'Dependents' 컬럼 기준으로 데이터 그룹화 및 unstack으로 형태 변경
partner_dependents = df.groupby(['Partner','Dependents']).size().unstack()

# 그룹화된 데이터를 바탕으로 바 그래프 생성
ax = (partner_dependents.T*100.0 / partner_dependents.T.sum()).T.plot(kind='bar',
                                                                      width = 0.2,
                                                                      stacked = True,
                                                                      rot = 0, 
                                                                      figsize = (8,6),
                                                                      color = colors)

# y축을 백분율로 표시
ax.yaxis.set_major_formatter(mtick.PercentFormatter())

# 범례 위치, 제목, 폰트 크기 설정
ax.legend(loc='center',prop={'size':14},title = 'Dependents',fontsize =14)

# y축 라벨 및 그래프 제목 설정
ax.set_ylabel('% Customers',size = 14)
ax.set_title('% Customers with/without dependents based on whether they have a partner',size = 14)

# x축 라벨의 폰트 크기 설정
ax.xaxis.label.set_size(14)

# 각 바 위에 백분율 값을 표시
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.0f}%'.format(height), (p.get_x()+.25*width, p.get_y()+.4*height),
                color = 'white',
               weight = 'bold',
               size = 14)


```

Output

![image](https://github.com/plintAn/Churn_EDA_Solution/assets/124107186/b12dd238-e653-44b0-9be0-a195bb15b148)

부양가족이 있는 경우 90% 비율로 '이탈 여부'가 'No'를 선택했고,파트너 여부에 따른 결과는 크게 차이가 없었다.

### 가설3

* 가설3.사용기간이 긴 고객들은 이탈여부가 적을 것이다.

```python
# df의 'tenure' 컬럼에 대한 히스토그램 생성. bin의 크기는 전체 기간을 5개월로 나눈 값으로 설정.
ax = sns.histplot(df['tenure'], bins=int(180/5), color='darkblue', edgecolor='black', linewidth=1)

# y축 라벨 설정
ax.set_ylabel('# of Customers')

# x축 라벨 설정
ax.set_xlabel('Tenure (months)')

# 그래프 제목 설정
ax.set_title('# of Customers by their tenure')
```

Output

![image](https://github.com/plintAn/Churn_EDA_Solution/assets/124107186/2d979ccb-6564-46f6-84da-72349a206fae)

'가입 기간'에 따른 고객 별 수는 신규 유입 고객과 70개월 이상인 고객이 두드러지게 많은 그래프를 보여주었다.



```python
import matplotlib.pyplot as plt
import seaborn as sns

# 1행 3열의 서브플롯을 생성 (각 계약 유형별로 'tenure'의 히스토그램을 그리기 위함)
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(20, 6))

# 월간 계약에 대한 히스토그램
sns.histplot(df[df['Contract'] == 'Month-to-month']['tenure'], 
             bins=int(180/5), color='turquoise', edgecolor='black',
             ax=ax1)
ax1.set_ylabel('# of Customers')  # y축 라벨 설정
ax1.set_xlabel('Tenure (months)')  # x축 라벨 설정
ax1.set_title('Month to Month Contract')  # 제목 설정

# 1년 계약에 대한 히스토그램
sns.histplot(df[df['Contract'] == 'One year']['tenure'], 
             bins=int(180/5), color='steelblue', edgecolor='black',
             ax=ax2)
ax2.set_xlabel('Tenure (months)')  # x축 라벨 설정
ax2.set_title('One Year Contract')  # 제목 설정

# 2년 계약에 대한 히스토그램
sns.histplot(df[df['Contract'] == 'Two year']['tenure'], 
             bins=int(180/5), color='darkblue', edgecolor='black',
             ax=ax3)
ax3.set_xlabel('Tenure (months)')  # x축 라벨 설정
ax3.set_title('Two Year Contract')  # 제목 설정

plt.show()  # 그래프 출력

```

Output

![image](https://github.com/plintAn/Churn_EDA_Solution/assets/124107186/dcb070d3-bbbb-4879-94ad-8624f7725a6f)


여기서 원-핫 인코딩을 진행한 'Month to Month Contract', 'One Year Contract', 'Two Year Contract' 에 대해 시각화를 진행. 각각 '이탈 여부'에 따른 월별 계약서 작성 시 음의 상관관계, 1년 단위 계약시 양의 상관관계, 2년 단위 계약시 강한 양의 상관관계를 보여준다.

## 2.EDA(탐색적 데이터 분석)

### '온라인 서비스'에 관한 '이탈 여부' 시각화

가장 처음 시각화한 그래프 확인 시 'OnlineSecurity_No' 와 같은 온라인 서비스를 이용하지 않은 고객들의 이탈률이 높았음에 기인하여 시각화를 진행.

여기서 참고해야할 점은, '온라인 서비스'에 가입한 사람에 대한 '이용 여부' 확인이므로 아예 가입조차 하지 않는다면 3번째 선택지인 'No service'로 선택하게 되었다.



```python
# 각종 서비스 항목 리스트
services = ['PhoneService','MultipleLines','InternetService','OnlineSecurity',
           'OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']

# 3x3의 서브플롯 생성 (총 9개 서비스 항목을 위한)
fig, axes = plt.subplots(nrows = 3, ncols = 3, figsize = (15, 12))

# 서비스 항목 각각에 대한 막대 그래프 생성
for i, item in enumerate(services):
    if i < 3:
        # 첫 번째 열에 대한 그래프
        ax = df[item].value_counts().plot(kind = 'bar', ax=axes[i, 0], rot = 0)
        
    elif i >= 3 and i < 6:
        # 두 번째 열에 대한 그래프
        ax = df[item].value_counts().plot(kind = 'bar', ax=axes[i-3, 1], rot = 0)
        
    elif i < 9:
        # 세 번째 열에 대한 그래프
        ax = df[item].value_counts().plot(kind = 'bar', ax=axes[i-6, 2], rot = 0)
    
    # 각 서브플롯의 제목 설정
    ax.set_title(item)

# 그래프 간의 간격 조절
plt.tight_layout()
# 그래프 출력
plt.show()
```



Output

![image](https://github.com/plintAn/Churn_EDA_Solution/assets/124107186/e43b3dcd-c8f3-4ee8-9887-a8b71d636d5b)

그래프 확인 결과,  'OnlineSecurity', 'TechSupport'  등 서비스를 이용하지 않은 고객이 더 많았다.


```python
# 사용할 색상 정의
colors = ['#4D3425','#E4512B']

# 'Churn' 칼럼의 값 별로 빈도를 계산하고, 전체 데이터 길이로 나누어 백분율을 구한 후, 막대 그래프로 표현
ax = (df['Churn'].value_counts()*100.0 /len(df)).plot(kind='bar',
                                                      stacked = True,
                                                      rot = 0,
                                                      color = colors,
                                                      figsize = (8,6))

# y축을 백분율 형식으로 설정
ax.yaxis.set_major_formatter(mtick.PercentFormatter())

# y축 라벨 설정
ax.set_ylabel('% Customers',size = 14)

# x축 라벨 설정
ax.set_xlabel('Churn',size = 14)

# 그래프 제목 설정
ax.set_title('Churn Rate', size = 14)

# 각 막대의 너비 값을 저장하기 위한 리스트 초기화
totals = []

# 각 막대의 너비 값을 리스트에 추가
for i in ax.patches:
    totals.append(i.get_width())

# 모든 막대의 너비 합계 계산
total = sum(totals)

# 각 막대 위에 백분율 값 표시
for i in ax.patches:
    ax.text(i.get_x()+.15, i.get_height()-4.0, \
            str(round((i.get_height()/total), 1))+'%',
            fontsize = 12,
            color='white',
            weight='bold')
```

Output

![image](https://github.com/plintAn/Churn_EDA_Solution/assets/124107186/16ce7547-343b-4e04-9d1a-0f06850a227c)

이용 고객 중 '이탈 여부'에 따른 비율 시각화. 데이터셋 고객의 73.4%가 현재 이용중인 고객으로 나타났다

'가입 기간' 별 이탈률 박스플롯

```python
sns.boxplot(x = df.Churn, y = df.tenure)
```

* 이탈 여부가 'No'인 고객은 15개월 ~ 60개월 사이 고르게 분포하고 있었고,
* 이탈 여부가 'Yes'인 고객은 1개월 ` 30개월 사이 앞쪽에 쏠린 분포를 보여주었다.

Output

![image](https://github.com/plintAn/Churn_EDA_Solution/assets/124107186/1d984a1e-4c9a-4757-b1e5-a4d5e0239561)


### 월별 요금과 '이탈 여부' 시각화

* 낮은 요금일 때 '이탈 여부'가 높고, '높은 요금'일 경우 '이탈 여부'가 크다

```python
# "Churn" 값이 'No'인 고객의 월별 요금에 대한 KDE 그래프를 빨간색으로 표시
ax = sns.kdeplot(df.MonthlyCharges[(df["Churn"] == 'No') ],
                color="Red", fill=True)

# "Churn" 값이 'Yes'인 고객의 월별 요금에 대한 KDE 그래프를 파란색으로 표시
ax = sns.kdeplot(df.MonthlyCharges[(df["Churn"] == 'Yes') ],
                ax=ax, color="Blue", fill=True)

# 범례 설정
ax.legend(["Not Churn", "Churn"], loc='upper right')

# y축 라벨 설정
ax.set_ylabel('Density')

# x축 라벨 설정
ax.set_xlabel('Monthly Charges')

# 그래프 제목 설정
ax.set_title('Distribution of monthly charges by churn')

```

이 KDE 그래프는 'MonthlyCharges' (월별 요금)에 따른 고객의 이탈률 분포를 시각적으로 보여준다

Output

![image](https://github.com/plintAn/Churn_EDA_Solution/assets/124107186/95425efd-9aaf-40b4-9ce5-2717f7b489e7)

그래프 KDE 밀도 분석 결과 '월별 요금'이 적은 경우 이탈률이 적었고, '월별 요금'이 60 ~ 110 인 경우 특히 이탈률이 높았다.


```python
# "Churn" 값이 'No'인 고객의 전체 요금에 대한 KDE 그래프를 빨간색으로 표시
ax = sns.kdeplot(df.TotalCharges[(df["Churn"] == 'No') ],
                color="Red", fill=True)

# "Churn" 값이 'Yes'인 고객의 전체 요금에 대한 KDE 그래프를 파란색으로 표시
ax = sns.kdeplot(df.TotalCharges[(df["Churn"] == 'Yes') ],
                ax=ax, color="Blue", fill=True)

# 범례 설정
ax.legend(["Not Churn", "Churn"], loc='upper right')

# y축 라벨 설정
ax.set_ylabel('Density')

# x축 라벨 설정
ax.set_xlabel('Total Charges')

# 그래프 제목 설정
ax.set_title('Distribution of total charges by churn')

```

이 KDE 그래프는 'Total Charges' (전체 요금)에 따른 고객의 이탈률 분포를 시각적으로 보여준다

Output

![image](https://github.com/plintAn/Churn_EDA_Solution/assets/124107186/9229f2eb-7a1b-4dd9-b34f-fbd34fd76065)

마찬가지로 '전체 요금' 역시 요금을 많이 내는 고객일수록 이탈률이 현저히 높았다.


</div>

<div id="2">

# II. 모델링

일단 타겟 변수 'Churn'을 제외하고 원-핫 인코딩을 실시한 설명 변수들을 라벨링

```python
# 'Churn' 열을 제외한 모든 열을 독립 변수로 지정합니다.
X = df_dummies.drop(columns=['Churn'])

# 종속 변수를 저장합니다.
y = df_dummies['Churn'].values
```

```python
# MinMaxScaler 클래스를 임포트합니다.
from sklearn.preprocessing import MinMaxScaler

# 스케일러를 생성합니다.
scaler = MinMaxScaler(feature_range=(0, 1))

# 독립 변수를 스케일링합니다.
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X))

# 'features' 변수를 정의합니다.
features = X.columns.tolist()

# 독립 변수의 이름을 다시 설정합니다.
X.columns = features
```

타겟 변수(종속 변수), 설명 변수(독립 변수) 라벨

```python
# sklearn의 train_test_split 모듈을 임포트합니다.
from sklearn.model_selection import train_test_split

# 데이터를 학습용과 테스트용으로 7:3 비율로 분리합니다. 랜덤 시드는 101로 설정합니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
```

```python
# sklearn에서 로지스틱 회귀 모델을 임포트합니다.
from sklearn.linear_model import LogisticRegression

# 로지스틱 회귀 모델 객체를 생성합니다.
model = LogisticRegression()

# 학습 데이터를 사용하여 모델을 학습시킵니다.
result = model.fit(X_train, y_train)
```

```python
# sklearn에서 로지스틱 회귀 모델을 임포트합니다.
from sklearn.linear_model import LogisticRegression

# 로지스틱 회귀 모델 객체를 생성합니다.
model = LogisticRegression()

# 학습 데이터를 사용하여 모델을 학습시킵니다.
result = model.fit(X_train, y_train)

```

```python

```

Output

```python
0.8075829383886256
```

```python
# 로지스틱 회귀 모델의 계수(가중치)를 판다스 시리즈로 변환합니다.
weights = pd.Series(model.coef_[0], index=X.columns.values)

# 가중치를 내림차순으로 정렬하고 상위 10개의 변수와 그 가중치를 막대 그래프로 출력합니다.
print(weights.sort_values(ascending=False)[:10].plot(kind='bar'))
```






## 4.모델 구축



## 5.모델 평가



## 6.모델 최적화






</div>

<div id="3">

# III. 결과 해석 및 적용

## 7.모델 해석



## 8.방안 제시


</div>

<div id="4">

IV. 배포 및 적용



## 9. 모델 배포





</div>

<div id="5">

</div>

<div id="6">

</div>

<div id="7">

</div>

<div id="8">

