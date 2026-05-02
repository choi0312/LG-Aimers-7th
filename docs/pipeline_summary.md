# Pipeline Summary

이 레포지토리는 세 개의 원본 제출 코드를 하나의 모듈형 구조로 통합한다.

## Pipeline 1

- 35-day sales history window
- calendar, weather, group, ski, room metadata
- LightGBM Tweedie multi-output regression
- variance power ensemble

## Pipeline 2

- Pipeline 1 계열 feature를 유지
- room zero-day correction 추가
- Hwadam-specific feature 추가
- weekpart-based extreme value clamp 추가

## Median Ensemble

- 두 제출 파일을 ID 기준으로 정렬
- common prediction columns에 대해 median
- non-negative integer clipping
