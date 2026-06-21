# 학생 생활패턴 기반 최종 성적 예측 및 학업 위험군 분류

## 프로젝트 개요
UCI Student Performance Dataset을 활용하여 학생의 생활패턴, 학교생활, 가정환경, 결석 정보, 과거 성적을 기반으로 최종 성적 G3를 예측하는 머신러닝 프로젝트입니다.

## 목표
- 회귀 모델을 활용한 최종 성적 G3 점수 예측
- 분류 모델을 활용한 학업 위험군 예측
- 모델 A와 모델 B의 성능 비교

## 모델 구성
- 모델 A: 생활패턴, 학교생활, 가정환경, 결석 정보 사용
- 모델 B: 모델 A 변수에 G1, G2 성적 추가

## 사용 모델
### 회귀
- Linear Regression
- KNN Regression

### 분류
- Logistic Regression
- KNN Classifier
- SVM

## 평가지표
### 회귀
- MAE
- MSE
- RMSE
- R2

### 분류
- Accuracy
- Precision
- Recall
- F1-score

## 주요 결과
G1, G2를 포함한 모델 B가 모델 A보다 전반적으로 높은 예측 성능을 보였습니다. 특히 최종 성적 G3는 1차 성적 G1, 2차 성적 G2와 높은 관련성을 가지는 것으로 확인되었습니다.

## 실행 방법
```bash
pip install pandas scikit-learn matplotlib seaborn streamlit
