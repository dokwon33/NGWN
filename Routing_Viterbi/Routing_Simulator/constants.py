"""
시뮬레이터 전역 물리 상수 모음
단위에 특히 주의!
1) 하나의 데이터 블록 크기를 64800으로 쓰고 있는데 이것도 상수화 필요 (하드코딩)
    from constants import BLOCK_SIZE
    self.size = BLOCK_SIZE 
    ※ transmission time 계산에서도 (buffer_size * 64800) / dataRate 이 부분 조심
2) 'MIN_ELEVATION_ANGLE'(satellite 기본 최소 elevation angle), 
    'DEFAULT_MOVEMENT_TIME'(기본 constellation 이동 주기[s]),
    'DEFAULT_GSL_DISTANCE'(GT coverage 관련 기본 거리 [km] 또는 맥락에 맞게 조정)
    => 상수화 시키는 것이 좋아보임!
"""

# Earth radius [m]
# 지구 반지름
# 위성-지상국 거리, 위성 궤도 반지름 계산에 사용
Re = 6378e3

# Universal gravitational constant [m^3 / (kg * s^2)]
# 만유인력 상수
# 위성 공전 주기 계산에 사용
G = 6.67259e-11

# Mass of Earth [kg]
# 지구 질량
# 중력 기반 궤도 주기 계산에 사용
Me = 5.9736e24

# Earth rotation period [s]
# 지구 자전 주기
# 항성일 기준으로 약 86164초
# constellation movement에서 지구 회전 반영할 때 사용
Te = 86164.28450576939

# Speed of light in vacuum [m/s]
# 광속
# propagation delay 계산에 사용
Vc = 299792458

# Boltzmann constant [J/K]
# 볼츠만 상수
# RF 링크의 열잡음(noise power) 계산에 사용
k = 1.38e-23

# Parabolic antenna efficiency [-]
# 포물면 안테나 효율
# 안테나 gain 계산에 사용
eff = 0.55

# Default data block size [bits]
# 시뮬레이터에서 하나의 데이터 블록 크기
# 여러 클래스에서 반복 사용되므로 상수화 권장
BLOCK_SIZE = 64800