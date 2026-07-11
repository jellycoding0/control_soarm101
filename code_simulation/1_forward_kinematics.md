# 정기구학(Forward Kinematics) 마스터하기
정기구학(Forward Kinematics)이란 **"로봇의 각 관절 각도($\theta$)가 주어졌을 때, 로봇 끝단(End-Effector)의 3차원 위치와 방향을 계산하는 과정"**을 말함. 기구학 연산의 가장 기본이 되는 단계로, 이를 수행하기 위해 우리는 관절과 링크 사이의 기하학적 관계를 정의하는 **DH 파라미터(Denavit-Hartenberg Parameters)**를 활용.

* **실습 코드 GitHub 저장소:** [jellycoding0/control_soarm101](https://github.com/jellycoding0/control_soarm101)
    - 실제 로봇 제어 코드 ->\code_simulation\1_forward_kinematics.py
    - 시뮬레이션 실습 코드 -> \code_soram\2_1_FK_관절공간_실습.py

## 1. DH 파라미터 설정 규칙
DH 파라미터를 설정하는 규칙만 정확히 준수한다면, 각 좌표계의 위치와 방향은 설계자가 자유롭게 설정할 수 있습니다. 규칙만 올바르게 따른다면 최종적으로 도출되는 해(정답)는 모두 동일하게 일치.

###  표준적인 좌표계 선정 가이드
1. **$z$축 선정:** 각 관절의 **회전축(또는 구동축)**을 $z$축으로 설정함.
2. **$x$축 선정:** 각 링크에 연결된 두 관절의 회전축($z_{i-1}$과 $z_i$)이 **서로 직교하거나 만나는 공통 수선 방향**으로 $x$축을 설정.

###  4가지 핵심 변수의 의미
* $\theta_i$ (Joint Angle): 관절의 회전각
* $d_i$ (Link Offset): 링크 변위 (관절 축 방향의 거리)
* $a_i$ (Link Length): 링크 길이 (두 관절 축 사이의 최단 거리)
* $\alpha_i$ (Link Twist): 링크 뒤틀림각 (두 관절 축이 어긋난 각도)

이 네 가지 변수를 구하기 위해 순서대로 **① $x_{i-1}$ 축에 대한 회전($z_i$ 통일) $\rightarrow$ ② $x_{i-1}$ 축에 대한 이동 $\rightarrow$ ③ $z_i$ 축에 대한 이동 $\rightarrow$ ④ $z_i$ 축에 대한 회전($x_i$ 통일)**의 과정을 거치게 됨.

### SO-ARM의 DH 파라미터 테이블 (예시)
앞서 확인한 SO-ARM의 실제 설계 데이터를 기반으로 구축된 DH 테이블임.

| Joint (관절) | $\alpha_{i-1}$ (Link Twist) | $a_{i-1}$ (Link Length) | $d_i$ (Link Offset) | $\theta_{offset}$ (Joint Offset) |
| :---: | :---: | :---: | :---: | :---: |
| **1** | $0.0$ | $0.000\,\text{m}$ | $0.119\,\text{m}$ | $0.0$ |
| **2** | $-\pi/2$ | $0.068\,\text{m}$ | $0.000\,\text{m}$ | $-\pi/2$ |
| **3** | $0.0$ | $0.111\,\text{m}$ | $0.000\,\text{m}$ | $0.0$ |
| **4** | $0.0$ | $0.137\,\text{m}$ | $0.000\,\text{m}$ | $-\pi/2$ |
| **5** | $-\pi/2$ | $0.000\,\text{m}$ | $0.099\,\text{m}$ | $0.0$ |
| **6** | $0.0$ | $0.000\,\text{m}$ | $0.060\,\text{m}$ | $0.0$ |

![ID 설정](../img/Juxi_Technology-SOARM101-3.jpg)


## 2. 변환 행렬(Transformation Matrix) 계산

설정된 DH 파라미터 값들을 활용하여, $i-1$번째 좌표계에서 $i$번째 좌표계로 변환해 주는 $4 \times 4$ 동차 변환 행렬(Homogeneous Transformation Matrix) $T_i$를 다음과 같이 유도함.

$$T_i = \begin{bmatrix} \cos(\theta_i) & -\sin(\theta_i)\cos(\alpha_i) & \sin(\theta_i)\sin(\alpha_i) & a_i\cos(\theta_i) \\ \sin(\theta_i) & \cos(\theta_i)\cos(\alpha_i) & -\cos(\theta_i)\sin(\alpha_i) & a_i\sin(\theta_i) \\ 0 & \sin(\alpha_i) & \cos(\alpha_i) & d_i \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

---

## 3. 정기구학(Forward Kinematics) 계산

SO-ARM과 같은 6자유도 로봇의 경우, 베이스(바닥)에서부터 최종 끝단(End-Effector)까지 연결된 모든 관절의 변환 행렬을 연쇄적으로 곱하여 로봇의 최종 위치를 구할 수 있습니다. 이는 다음과 같은 행렬의 곱으로 표현됨.

$$T_{\text{end-effector}} = T_1 \cdot T_2 \cdot T_3 \cdot T_4 \cdot T_5 \cdot T_6$$

여기서 각 $T_i$는 해당 관절의 DH 파라미터(테이블에 명시된 값)에 따라 고유하게 정의되는 개별 변환 행렬임.

### 💡 계산 예시: 첫 번째 관절의 변환 행렬 ($T_1$)
예를 들어, 테이블의 1번 행 데이터($\alpha_0=0.0$, $a_0=0.000$, $d_1=0.119$)와 첫 번째 관절의 보정된 각도 $\theta_1$을 매칭하여 계산식에 대입하면 $T_1$은 삼각함수 성질($\cos(0)=1, \sin(0)=0$)에 의해 다음과 같이 간단하게 정리됨.

$$T_1 = \begin{bmatrix} \cos(\theta_1) & -\sin(\theta_1) & 0 & 0 \\ \sin(\theta_1) & \cos(\theta_1) & 0 & 0 \\ 0 & 0 & 1 & 0.119 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

이와 같은 방식으로 각 관절마다 $T_2, T_3, \dots, T_6$를 테이블 값을 이용해 순서대로 구한 뒤 모두 곱하면 정기구학 연산이 완료됨. 이를 통해 최종적으로 로봇 끝단(End-Effector)이 3차원 공간에서 가리키는 정확한 **위치(Position, $P$)**와 **방향(Orientation, $R$)** 정보를 얻게 됨.

$$T_{\text{end-effector}} = \begin{bmatrix} R_{11} & R_{12} & R_{13} & P_x \\ R_{21} & R_{22} & R_{23} & P_y \\ R_{31} & R_{32} & R_{33} & P_z \\ 0 & 0 & 0 & 1 \end{bmatrix}$$