# 운동 동작 분류 AI 경진대회 - 데이콘

![](https://dacon.s3.ap-northeast-2.amazonaws.com/competition/235689/header_background.jpeg)

* 아이디어 설명: https://www.notion.so/AI-54065ab4da62480f9ea595373c6f6454
* dacon 소스코드 요약본: https://dacon.io/competitions/official/235689

2021.01.11 ~ 2021.02.22  
public 18위 0.52696, private 21위 0.62167

61가지 운동 동작을 수행하면서 몸에 부착된 센서로부터 들어오는 6가지 입력(가속도 센서 + 자이로 센서)을 통해 어떤 운동 동작인지 분류한다.

데이터의 수가 적으면서(train에 3100여개) 극단적으로 치우친 데이터셋(3100개 중 약 1500개가 하나의 class이고 나머지 class들도 각각 2배 이상 개수 차이가 있기도 함)

초반에 일정 스코어를 달성한 이후 뭘 해도 성능 향상이 잘 안되어서 특히 상위권의 솔루션이 궁금했다.

## 시도해본 것

### 네트워크 모델링

- ResNet1d  
모든 conv, batchnorm, pooling 등을 1d로 변환해서 사용  
특이한 점은 pooling layer를 제거했을 때가 오히려 성능이 좋았다는 점?
- ResNeSt
- CNN + Transformer + CNN  
기본적으로 Transformer를 사용해서 큰 성능 향상을 볼 수 있었음.
- Channel Attention + Transformer + CNN - 약간 더 성능 향상.
- [Non-local Neural Network (CVPR 2018)](https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.html)  
Transformer가 spatial attention을 주로 담당하고 있는 것으로 보고 Transformer를 대체해보려는 시도... 생각보다 성능이 좋지는 않았음

### Optimization

- Adam, RAdam, AdamW 등등  
원래는 개인적인 취향으로 RAdam을 제일 많이 써왔는데, 이번에는 AdamW가 공통적으로 높은 성능을 보이는 것을 확인
- [SAM(Sharpness-Aware Minimization for Efficiently Improving Generalization) (Google, 2021)](https://arxiv.org/abs/2010.01412)  
optimizer를 추가로 최적화 하는 방법으로 최근 imagenet SOTA. 대신 epoch당 학습 속도는 1.8배정도 느려짐  
간단히 추가하는 것만으로도 성능 향상이 어느정도 보장이 되다 보니 많은 대회나 논문에서 사용할 것으로 보인다.
- [FocalLoss (CVPR 2017)](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html)  
데이터의 치우침이 크기 때문에 gamma를 약간 키워서(기본 2.0 --> 3.2정도) 쓰는게 좋았다.
- [ClassBalancedLoss (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html)  
오히려 치우침에 무관하게 학습되기 때문에 가장 수가 많은 Non-Exercise의 예측 정확도가 떨어지는 문제가 있었다.

### Augmentation

- Standardization
- Random Shifting  
몇몇 클래스에서 약간 반복되는 성질을 띄기 때문에 성능 향상이 있던 것으로 보임
- Additive waves??  
랜덤한 frequency와 amplitude의 sin/cos파를 더해줘서 기존과 비슷한 모양
- Random Blur  
랜덤하게 gaussian 필터 적용
- Test Time Augmentation  
- 파생 변수  
  - 임의로 생성해는 속도/위치 데이터
  - xyz +yaw pitch roll 축을 합쳐서 만든 총 가속도
- Oversampling + Undersampling
