# SEO_Bayesian

# 📌 SEO 검색 의도 분석 프로젝트 (Bayesian Optimization for Search Intent Analysis)

이 프로젝트는 **베이지안 접근법(Bayesian Optimization)**을 활용하여 **SEO(Search Engine Optimization) 최적화** 및 **검색 의도 분석(Search Intent Analysis)**을 수행하는 머신러닝 모델을 구축하는 것을 목표로 합니다. 

GitHub 저장소: [SEO_Bayesian](https://github.com/seonmin8284/SEO_Bayesian)

## **🚀 프로젝트 개요**

검색 의도(Search Intent)는 사용자가 특정 검색어를 입력했을 때 **정보 탐색(Informational), 비교 쇼핑(Commercial), 구매(Transactional)** 중 어떤 의도를 가지고 있는지를 분석하는 과정입니다. 

이 프로젝트에서는 **사용자의 검색 행동 데이터(조회, 장바구니 추가, 구매 등)를 바탕으로 베이지안 머신러닝 모델을 적용**하여 검색 의도를 자동 분류하는 시스템을 구축합니다.

## **🔍 데이터 개요**
### **사용된 데이터셋**
이 프로젝트에서는 **전자상거래(E-Commerce)에서 수집된 사용자 이벤트 데이터**를 활용합니다.

📌 **사용된 주요 파일:**
- `events.csv` → 사용자 행동 데이터 (조회, 장바구니 추가, 구매 등)
- `item_properties_part1.csv`, `item_properties_part2.csv` → 제품 속성 데이터
- `category_tree.csv` → 제품 카테고리 정보
- `searchOpt-checkpoint.ipynb` → 프로젝트 코드가 포함된 Jupyter Notebook

### **데이터 구성 요소**
| 컬럼명 | 설명 |
|--------|----------------------------------------|
| `timestamp` | 이벤트 발생 시간 (Unix Time) |
| `event` | 사용자 행동 (view, addtocart, transaction) |
| `itemid` | 제품 고유 ID |
| `property` | 제품 속성 (예: 크기, 색상, 브랜드 등) |
| `value` | 제품 속성 값 |

## **🛠️ 프로젝트 개념 및 논리**
### **1️⃣ 검색 의도 분류(Search Intent Classification)**
사용자의 행동 데이터를 바탕으로 검색 의도를 다음과 같이 분류합니다.

- **Informational (정보 탐색)**: 제품을 조회(view)하지만 장바구니에 추가하거나 구매하지 않음
- **Commercial (비교 쇼핑)**: 제품을 장바구니(addtocart)에 추가하지만 구매하지 않음
- **Transactional (구매 의도)**: 제품을 실제로 구매(transaction)

### **2️⃣ 데이터 처리 및 전처리**
- `events.csv`와 `item_properties.csv` 데이터를 **병합하지 않고 필요할 때만 조회**하여 메모리 사용 최적화
- 제품 속성 데이터를 ID 숫자가 아닌 **실제 의미 있는 단어(예: 브랜드, 카테고리, 크기 등)로 변환**
- 검색어를 **TF-IDF 벡터화**하여 머신러닝 모델에 입력할 수 있도록 변환

### **3️⃣ 베이지안 모델을 활용한 검색 의도 분석**
**TF-IDF 벡터화** 후, **Complement Naive Bayes**를 사용하여 검색 의도를 분류합니다.

```python
# TF-IDF 벡터화 및 Naive Bayes 모델 적용
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=3, max_df=0.9)),
    ("nb", ComplementNB())  # MultinomialNB보다 클래스 불균형에 강함
])
```

## **📊 모델 평가 및 예측 결과**
### **🔹 모델 평가 지표**
- `accuracy_score` (정확도)
- `classification_report` (Precision, Recall, F1-score 등)

```python
# 예측 및 평가
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

### **🔹 일부 예측 결과 예시**
| 검색어 데이터 | 예측된 검색 의도 |
|-------------|----------------|
| 'iphone case blue' | Commercial |
| 'best budget laptop 2024' | Informational |
| 'buy macbook pro' | Transactional |

## **📌 주요 개선점 및 향후 과제**
✅ **데이터 품질 개선**: 제품 속성(property) ID를 사람이 이해할 수 있는 의미로 변환 
✅ **실제 검색어 반영**: 실제 키워드 기반의 자연어 데이터를 학습하여 의미 있는 검색어 분석 수행
✅ **베이지안 최적화(Bayesian Optimization)**: 하이퍼파라미터 튜닝을 자동화하여 더 높은 정확도 달성

---
