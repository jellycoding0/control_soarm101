# 자코비안(Jacobian)과 특이점(Singularity) 마스터하기

정기구학을 통해 로봇의 '위치'를 구했다면, 이제 로봇이 '얼마나 빠른 속도로 움직이는지' 연산할 차례입니다. 이를 위해 필요한 개념이 바로 **자코비안(Jacobian)**과 기구학적 한계를 뜻하는 **특이점(Singularity)**입니다.

* **자코비안(Jacobian):** 로봇의 관절 속도($\dot{\theta}$)와 끝단(End-Effector)의 선속도/각속도($v, \omega$) 사이의 선형 사상 관계를 나타내는 행렬입니다.
* **특이점(Singularity):** 자코비안 행렬의 행렬식(Determinant)이 $0$이 되는 지점으로, 이 순간 로봇은 특정 방향으로의 자유도를 상실하여 움직일 수 없는 상태에 빠지게 됩니다.

---

## 1. 2자유도 평면 로봇의 자코비안 행렬 유도

가장 단순한 2자유도(2-DOF) 평면 링크 기구를 예시로 자코비안 행렬이 유도되는 과정을 살펴보겠습니다.

<img src="../img/시뮬_자코비안계산.png" width="300" alt="자코비안 계산 예시">

### 1) 정기구학 위치 방정식
로봇 끝단의 위치 좌표 $p_x, p_y$는 링크 길이($L_1, L_2$)와 관절 각도($\theta_1, \theta_2$)를 이용해 다음과 같이 정의됩니다.

$$p_x = L_1 \cos\theta_1 + L_2 \cos(\theta_1 + \theta_2)$$
$$p_y = L_1 \sin\theta_1 + L_2 \sin(\theta_1 + \theta_2)$$

### 2) 시간에 대한 미분 (속도 관계식 유도)
위 위치 방정식을 시간 $t$에 대해 미분(연쇄 법칙 적용)하면, 끝단의 속도 $v_x, v_y$와 관절 속도 $\dot{\theta}_1, \dot{\theta}_2$의 관계식을 얻을 수 있습니다.

$$v_x = \dot{p}_x = -\big(L_1 \sin\theta_1 + L_2 \sin(\theta_1 + \theta_2)\big)\dot{\theta}_1 - L_2 \sin(\theta_1 + \theta_2)\dot{\theta}_2$$
$$v_y = \dot{p}_y = \big(L_1 \cos\theta_1 + L_2 \cos(\theta_1 + \theta_2)\big)\dot{\theta}_1 + L_2 \cos(\theta_1 + \theta_2)\dot{\theta}_2$$

### 3) 행렬 형태로 변환 (Jacobian Matrix 도출)
위의 연립 방정식을 행렬 형태로 묶어내면 최종적으로 우리가 원하는 자코비안 행렬 $J_v$가 정의됩니다.

$$\begin{bmatrix} v_x \\ v_y \end{bmatrix} = \begin{bmatrix} -L_1 \sin\theta_1 - L_2 \sin(\theta_1 + \theta_2) & -L_2 \sin(\theta_1 + \theta_2) \\ L_1 \cos\theta_1 + L_2 \cos(\theta_1 + \theta_2) & L_2 \cos(\theta_1 + \theta_2) \end{bmatrix} \begin{bmatrix} \dot{\theta}_1 \\ \dot{\theta}_2 \end{bmatrix}$$

$$J_v = \begin{bmatrix} -L_1 \sin\theta_1 - L_2 \sin(\theta_1 + \theta_2) & -L_2 \sin(\theta_1 + \theta_2) \\ L_1 \cos\theta_1 + L_2 \cos(\theta_1 + \theta_2) & L_2 \cos(\theta_1 + \theta_2) \end{bmatrix}$$

---

## 2. 특이점 (Singularity)의 수학적 의미

<img src="../img/시뮬_특이점.png" width="450" alt="특이점 예시">

특이점은 로봇의 끝단이 물리적/기구학적 한계에 도달하여 특정 방향으로 더 이상 움직일 수 없는 자세를 의미합니다. (예: 팔을 완전히 일직선으로 쭉 편 상태)

###  특이점의 주요 특징
* **$\det(J) = 0$**: 자코비안 행렬의 행렬식(Determinant) 값이 $0$이 됩니다.
* **역행렬 존재성 상실**: 역기구학 속도를 구할 때 필요한 자코비안의 역행렬($J^{-1}$)이 존재하지 않게 됩니다.
* **자유도 상실 (Rank Loss)**: 자코비안 행렬의 계수(Rank)를 1개 이상 잃게 되며, 이는 로봇이 특정 축 방향으로 제어력을 잃어버림을 뜻합니다.

###  행렬식 계산 공식 레퍼런스
$2 \times 2$ 행렬의 행렬식은 다음과 같이 계산됩니다.

$$\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$$

위 예시에서 구한 자코비안 행렬 $J_v$의 행렬식을 구해 보면 다음과 같이 단순화됩니다.

$$\det(J_v) = L_1 L_2 \sin\theta_2$$

즉, $\sin\theta_2 = 0$이 되는 지점, 다시 말해 **$\theta_2 = 0^\circ$(팔을 쭉 편 상태)이거나 $\theta_2 = 180^\circ$(팔이 완전히 안으로 접힌 상태)일 때** 로봇은 특이점에 빠지게 됩니다.


## 3. [실습] 파이썬 자코비안 & 특이점 시뮬레이터 구현
로봇의 관절 각도를 조절하며 특이점 상태를 실시간으로 확인하는 파이썬 시뮬레이터 코드를 작성해 봅시다.
2_1_jacobian_singuarity.py 참고


##참고)
이 다음 SO-ARM의 역기구학 계산시 위의 방법처럼 행렬을 구하진 않고
컴퓨터 수치해석 방식으로 관절들을 미세하게 움직여보며 끝단의 실시간 움직임 변화량을 측정
이 변화량을 바탕으로 6행 5열짜리 자코비안 행렬($J$)을 만들고, 
DLS(Damped Least Squares) 역기구학 알고리즘 계산함 -> np.linalg.solve(H, J_T @ error_vector)