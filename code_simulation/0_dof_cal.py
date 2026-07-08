# 자유도는 로보틱스 신입 면접 빈출 질문

def calculate_dof(num_links, num_joints, joint_dofs):
    m = 6 # 3D 공간에서 링크가 가질 수 있는 최대 자유도는 6
    # SO-ARM -> LINK 6개 | Joint 5개
    # 대부분의 모터(회전 모터, 리니어 모터 등)는 joint_dofs = 1
    # 구면 조인트(Ball joint) 같은 특수 기구 구조는 joint_dofs = 2 또는 3
    F = m * (num_links - num_joints - 1) + sum(joint_dofs) # Kutzbach Formula 적용
    return F


print("자유도 계산기")

# 사용자 입력 받기
num_links = int(input("링크의 개수를 입력하세요: "))
num_joints = int(input("조인트의 개수를 입력하세요: "))

# 특수 모터가 있는경우에만 각 관절이 가지는 자유도 입력 받기
has_special_joint = str(input("Ball joint같은 특수모터가 있나요? (Y/N): ")).strip().upper()

if has_special_joint == "N":
    # 모든 조인트의 자유도가 1일 때, 조인트 개수만큼 [1, 1, 1...] 리스트 생성
    joint_dofs = [1] * num_joints
else:
    joint_dofs = []
    for i in range(num_joints):
        dof = int(input(f"{i + 1}번 관절의 자유도를 입력하세요: "))
        joint_dofs.append(dof)
        
# 자유도 계산
total_dof = calculate_dof(num_links, num_joints, joint_dofs)
print(f"계산된 로봇 자유도: {total_dof}")

