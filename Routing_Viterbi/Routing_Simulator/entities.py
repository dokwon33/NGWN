import os
import time
import math

import simpy
import geopy.distance
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from PIL import Image
from scipy.optimize import linear_sum_assignment

from constants import Re, G, Me, Te, Vc, BLOCK_SIZE
from models import RFlink
from routing import createGraph, getShortestPath
from sim_state import (
    receivedDataBlocks,
    createdBlocks,
    upGSLRates,
    downGSLRates,
    interRates,
    intraRate,
)
# Entities.py
## 지구, gateway, 위성, 데이터블록 같은 시뮬레이션의 상턔(state)를 가진 객체 정의
"""
Earth
 ├─ gateways: list[Gateway]
 ├─ cells: 2D list[Cell]
 └─ LEO: list[OrbitalPlane]
       └─ sats: list[Satellite]

Gateway
 ├─ sendBuffer
 ├─ paths
 └─ linkedSat

Satellite
 ├─ sendBufferGT
 ├─ sendBufferSatsIntra
 ├─ sendBufferSatsInter
 ├─ intraSats
 ├─ interSats
 └─ linkedGT

DataBlock
 ├─ source
 ├─ destination
 ├─ path
 └─ latency info

 # Workflow
    Earth 생성
    ├─ Cell들 생성
    ├─ Gateway들 생성
    └─ OrbitalPlane들 생성
        └─ Satellite들 생성

    Earth.linkCells2GTs()
    → 각 Cell이 가까운 Gateway에 연결

    Earth.linkSats2GTs()
    → 각 Gateway가 하나의 Satellite와 연결

    Gateway.fillBlock()
    → DataBlock 생성
    → path 붙임
    → Gateway.sendBuffer에 적재

    Gateway.sendBlock()
    → linked satellite로 uplink 전송

    Satellite.receiveBlock()
    → next hop 확인 후 satellite queue 또는 GT queue에 적재

    Satellite.sendBlock()
    → 다음 위성/GT로 전달

    Gateway.receiveBlock()
    → receivedDataBlocks에 저장
 """
class OrbitalPlane:
    """
    OrbitalPlane : 하나의 궤도면(orbit plane)
    - 궤도면의 고도, 경사각, 경도 저장
    - 궤도면에 속한 위성들 생성
    - 시간이 지나면 위성들을 회전
    self.h: 고도 
    self.longitude: 궤도면 경도
    self.inclination: 궤도 경사각
    self.sats: 궤도면에 속한 위성 리스트
    self.period: 공전 주기

    rotate(delta_t) => 시간이 delta_t만큼 흐르면 '지구 자전+궤도 이동'을 반영해서 위성 위치 갱신
    """
    def __init__(self, ID, h, longitude, inclination, n_sat, min_elev, firstID, env):
        self.ID = ID
        self.h = h
        self.longitude = longitude
        self.inclination = math.pi / 2 - inclination
        self.n_sat = n_sat
        self.period = 2 * math.pi * math.sqrt((self.h + Re) ** 3 / (G * Me))
        self.v = 2 * math.pi * (h + Re) / self.period
        self.min_elev = math.radians(min_elev)
        self.max_alpha = math.acos(Re * math.cos(self.min_elev) / (self.h + Re)) - self.min_elev
        self.max_beta = math.pi / 2 - self.max_alpha - self.min_elev
        self.max_distance_2_ground = Re * math.sin(self.max_alpha) / math.sin(self.max_beta)

        self.first_sat_ID = firstID
        self.sats = []
        for i in range(n_sat):
            self.sats.append(
                Satellite(
                    self.first_sat_ID + str(i),
                    int(self.ID),
                    int(i),
                    self.h,
                    self.longitude,
                    self.inclination,
                    self.n_sat,
                    env,
                )
            )

        self.last_sat_ID = self.first_sat_ID + str(len(self.sats) - 1)

    def __repr__(self):
        return (
            f"\nID = {self.ID}\n altitude= {self.h/1e3} km\n"
            f" longitude= {math.degrees(self.longitude):.2f} deg\n"
            f" inclination= {math.degrees(self.inclination):.2f} deg\n"
            f" number of satellites= {self.n_sat:.2f}\n"
            f" period= {self.period/3600:.2f} hours\n"
            f" satellite speed= {self.v/1e3:.2f} km/s"
        )

    def rotate(self, delta_t):
        self.longitude = self.longitude + 2 * math.pi * delta_t / Te
        self.longitude = self.longitude % (2 * math.pi)

        for sat in self.sats:
            sat.rotate(delta_t, self.longitude, self.period)


class Satellite:
    """
    Satellite: 실제로 패킷을 받고, 큐에 넣고, 다음 hop으로 보내는 라우터 역할
    - 현재 위성 위치 저장
    - 인접 위성(intra/inter plane) 저장
    - 연결된 게이트웨이 저장
    - 각 링크 별 송신 버펴(queue) 관리
    - 데이터 블록 수신/전송 처리
    
    (위치 관련 attribute)
    1. x,y,z
    2. latitude, longitude
    3. in_plane, i_in_plane (어느 궤도면의 몇 번째 위성인지)

    (링크 관련 attribute)
    1. linkedGT: 현재 연결된 gateway
    2. intraSats : 같은 궤도면 이웃 위성
    3. interSats : 다른 궤도면 이웃 위성

    (버퍼 관련 attribute)
    * 위성은 링크마다 큐를 따로 갖는 특성
    1. sendBufferGT: GT로 내려보내는 큐(1개)
    2. sendBufferSatsIntra: intra-plane 링크별 큐
    3. sendBufferSatsInter: inter-plane 링크별 큐

    receiveBlock(block, propTime) => 위성이 블록을 받았을 때 호출 
    : 수신 처리 담당
    * Workflow
    ** 1. prop. delay만큼 wait
    ** 2. 블록의 path 확인
    ** 3. 다음 hop이 GT인지, 다른 위성인지 확인
    ** 4. 그에 맞는 버퍼에 블록을 insert

    sendBlock(destination, isSat, isIntra=None) => 위성이 버퍼에서 꺼낼 때 실제로 전송하는
    Simpy process
    : 송신 처리 담당
    * Workflow
    ** 1. 큐에 블록이 들어올 때까지 기다림
    ** 2. transmission time 계산
    ** 3. transmission delay만큼 기다림
    ** 4. receiver 쪽 receiveBlock() 시작
    ** 5. 큐에서 블록 제거

    adjustDownRate() => 위성->GT downlink 속도 계산
    : SNR과 modulation efficiency threshold 이용해서 rate 정하는 구조
    : 물리계층 링크 용량 계산

    getLinkLatencies(graph) => 현재 위성이 가진 각 outgoing 링크에 대해 latency 계산
    * 현재 latency = propagation delay + transmission delay 
    *** path metric이 'latency'일 때 중요
    """
    def __init__(self, ID, in_plane, i_in_plane, h, longitude, inclination, n_sat, env, quota=500, power=10):
        self.ID = ID
        self.in_plane = in_plane
        self.i_in_plane = i_in_plane
        self.quota = quota
        self.h = h
        self.power = power
        self.minElevationAngle = 30

        self.r = Re + self.h
        self.theta = 2 * math.pi * self.i_in_plane / n_sat
        self.phi = longitude
        self.inclination = inclination

        self.x = self.r * (
            math.sin(self.theta) * math.cos(self.phi)
            - math.cos(self.theta) * math.sin(self.phi) * math.sin(self.inclination)
        )
        self.y = self.r * (
            math.sin(self.theta) * math.sin(self.phi)
            + math.cos(self.theta) * math.cos(self.phi) * math.sin(self.inclination)
        )
        self.z = self.r * math.cos(self.theta) * math.cos(self.inclination)

        self.polar_angle = self.theta
        self.latitude = math.asin(self.z / self.r)

        if self.x > 0:
            self.longitude = math.atan(self.y / self.x)
        elif self.x < 0 and self.y >= 0:
            self.longitude = math.pi + math.atan(self.y / self.x)
        elif self.x < 0 and self.y < 0:
            self.longitude = math.atan(self.y / self.x) - math.pi
        elif self.y > 0:
            self.longitude = math.pi / 2
        elif self.y < 0:
            self.longitude = -math.pi / 2
        else:
            self.longitude = 0

        self.waiting_list = {}
        self.applications = []
        self.n_sat = n_sat

        f = 20e9
        B = 500e6
        maxPtx = 10
        Adtx = 0.26
        Adrx = 0.33
        pL = 0.3
        Nf = 1.5
        Tn = 50
        min_rate = 10e3
        self.ngeo2gt = RFlink(f, B, maxPtx, Adtx, Adrx, pL, Nf, Tn, min_rate)
        self.downRate = 0

        self.env = env
        self.sendBufferGT = ([env.event()], [])
        self.sendBlocksGT = []
        self.sats = []
        self.linkedGT = None
        self.GTDist = None
        self.tempBlocks = []

        self.intraSats = []
        self.interSats = []
        self.sendBufferSatsIntra = []
        self.sendBufferSatsInter = []
        self.sendBlocksSatsIntra = []
        self.sendBlocksSatsInter = []
        self.newBuffer = [False]

    def maxSlantRange(self):
        eps = math.radians(self.minElevationAngle)
        distance = math.sqrt((Re + self.h) ** 2 - (Re * math.cos(eps)) ** 2) - Re * math.sin(eps)
        return distance

    def __repr__(self):
        return (
            f"\nID = {self.ID}\n orbital plane= {self.in_plane}, index in plane= {self.i_in_plane}, h={self.h:.2f}\n"
            f" pos r = {self.r:.2f}, pos theta = {self.theta:.2f}, pos phi = {self.phi:.2f},\n"
            f" pos x= {self.x:.2f}, pos y= {self.y:.2f}, pos z= {self.z:.2f}\n"
            f" inclination = {math.degrees(self.inclination):.2f}\n"
            f" polar angle = {math.degrees(self.polar_angle):.2f}\n"
            f" latitude = {math.degrees(self.latitude):.2f}\n"
            f" longitude = {math.degrees(self.longitude):.2f}"
        )

    def createReceiveBlockProcess(self, block, propTime):
        self.env.process(self.receiveBlock(block, propTime))

    def receiveBlock(self, block, propTime):
        self.tempBlocks.append(block)

        yield self.env.timeout(propTime)

        if block.path == -1:
            return

        block.propLatency += propTime

        for i, tempBlock in enumerate(self.tempBlocks):
            if block.ID == tempBlock.ID:
                self.tempBlocks.pop(i)
                break

        block.checkPoints.append(self.env.now)

        index = None
        for i, step in enumerate(block.path):
            if self.ID == step[0]:
                index = i

        if index == len(block.path) - 2:
            if not self.sendBufferGT[0][0].triggered:
                self.sendBufferGT[0][0].succeed()
                self.sendBufferGT[1].append(block)
            else:
                newEvent = self.env.event().succeed()
                self.sendBufferGT[0].append(newEvent)
                self.sendBufferGT[1].append(block)
        else:
            ID = None
            isIntra = False

            for sat in self.intraSats:
                if sat[1].ID == block.path[index + 1][0]:
                    ID = sat[1].ID
                    isIntra = True

            for sat in self.interSats:
                if sat[1].ID == block.path[index + 1][0]:
                    ID = sat[1].ID

            if ID is not None:
                sendBuffer = None
                if isIntra:
                    for buffer in self.sendBufferSatsIntra:
                        if ID == buffer[2]:
                            sendBuffer = buffer
                else:
                    for buffer in self.sendBufferSatsInter:
                        if ID == buffer[2]:
                            sendBuffer = buffer

                if not sendBuffer[0][0].triggered:
                    sendBuffer[0][0].succeed()
                    sendBuffer[1].append(block)
                else:
                    newEvent = self.env.event().succeed()
                    sendBuffer[0].append(newEvent)
                    sendBuffer[1].append(block)
            else:
                print(
                    f"ERROR! Sat {self.ID} tried to send block to {block.path[index + 1][0]} "
                    f"but did not have it in its linked satellite list"
                )
                print(block.path)
                for neighbor in self.interSats:
                    print(neighbor[1].ID)
                for neighbor in self.intraSats:
                    print(neighbor[1].ID)
                print(block.isNewPath)
                print(block.oldPath)
                print(block.newPath)
                raise RuntimeError("Next satellite buffer not found")

    def sendBlock(self, destination, isSat, isIntra=None):
        if isIntra is not None:
            sendBuffer = None
            if isSat:
                if isIntra:
                    for buffer in self.sendBufferSatsIntra:
                        if buffer[2] == destination[1].ID:
                            sendBuffer = buffer
                else:
                    for buffer in self.sendBufferSatsInter:
                        if buffer[2] == destination[1].ID:
                            sendBuffer = buffer
        else:
            sendBuffer = self.sendBufferGT

        while True:
            try:
                yield sendBuffer[0][0]

                sendBuffer[1][0].checkPointsSend.append(self.env.now)

                if isSat:
                    timeToSend = sendBuffer[1][0].size / destination[2]
                    propTime = self.timeToSend(destination)
                    yield self.env.timeout(timeToSend)
                    receiver = destination[1]
                else:
                    propTime = self.timeToSend(self.linkedGT.linkedSat)
                    timeToSend = sendBuffer[1][0].size / self.downRate
                    yield self.env.timeout(timeToSend)
                    receiver = self.linkedGT

                if True in self.newBuffer and not isIntra and isSat:
                    if isIntra is not None:
                        sendBuffer = None
                        if isSat:
                            if isIntra:
                                for buffer in self.sendBufferSatsIntra:
                                    if buffer[2] == destination[1].ID:
                                        sendBuffer = buffer
                            else:
                                for buffer in self.sendBufferSatsInter:
                                    if buffer[2] == destination[1].ID:
                                        sendBuffer = buffer
                    else:
                        sendBuffer = self.sendBufferGT

                    for index, val in enumerate(self.newBuffer):
                        if val:
                            self.newBuffer[index] = False
                            break

                sendBuffer[1][0].txLatency += timeToSend
                receiver.createReceiveBlockProcess(sendBuffer[1][0], propTime)

                if len(sendBuffer[0]) == 1:
                    sendBuffer[0].pop(0)
                    sendBuffer[1].pop(0)
                    sendBuffer[0].append(self.env.event())
                else:
                    sendBuffer[0].pop(0)
                    sendBuffer[1].pop(0)

            except simpy.Interrupt:
                break

    def adjustDownRate(self):
        speff_thresholds = np.array(
            [0, 0.434841, 0.490243, 0.567805, 0.656448, 0.789412, 0.889135, 0.988858,
             1.088581, 1.188304, 1.322253, 1.487473, 1.587196, 1.647211, 1.713601,
             1.779991, 1.972253, 2.10485, 2.193247, 2.370043, 2.458441, 2.524739,
             2.635236, 2.637201, 2.745734, 2.856231, 2.966728, 3.077225, 3.165623,
             3.289502, 3.300184, 3.510192, 3.620536, 3.703295, 3.841226, 3.951571,
             4.206428, 4.338659, 4.603122, 4.735354, 4.933701, 5.06569, 5.241514,
             5.417338, 5.593162, 5.768987, 5.900855]
        )
        lin_thresholds = np.array(
            [1e-10, 0.5188000389, 0.5821032178, 0.6266138647, 0.751622894, 0.9332543008,
             1.051961874, 1.258925412, 1.396368361, 1.671090614, 2.041737945, 2.529297996,
             2.937649652, 2.971666032, 3.25836701, 3.548133892, 3.953666201, 4.518559444,
             4.83058802, 5.508076964, 6.45654229, 6.886522963, 6.966265141, 7.888601176,
             8.452788452, 9.354056741, 10.49542429, 11.61448614, 12.67651866, 12.88249552,
             14.48771854, 14.96235656, 16.48162392, 18.74994508, 20.18366364, 23.1206479,
             25.00345362, 30.26913428, 35.2370871, 38.63669771, 45.18559444, 49.88844875,
             52.96634439, 64.5654229, 72.27698036, 76.55966069, 90.57326009]
        )

        pathLoss = 10 * np.log10((4 * math.pi * self.linkedGT.linkedSat[0] * self.ngeo2gt.f / Vc) ** 2)
        snr = 10 ** ((self.ngeo2gt.maxPtx_db + self.ngeo2gt.G - pathLoss - self.ngeo2gt.No) / 10)

        feasible_speffs = speff_thresholds[np.nonzero(lin_thresholds <= snr)]
        speff = self.ngeo2gt.B * feasible_speffs[-1]
        self.downRate = speff

    def timeToSend(self, linkedSat):
        distance = linkedSat[0]
        return distance / Vc

    def findNeighbours(self, earth):
        self.linked = None
        self.upper = earth.LEO[self.in_plane].sats[self.i_in_plane - 1]
        if self.i_in_plane < self.n_sat - 1:
            self.lower = earth.LEO[self.in_plane].sats[self.i_in_plane + 1]
        else:
            self.lower = earth.LEO[self.in_plane].sats[0]

    def rotate(self, delta_t, longitude, period):
        self.phi = longitude
        self.theta = self.theta + 2 * math.pi * delta_t / period
        self.theta = self.theta % (2 * math.pi)

        self.x = self.r * (
            math.sin(self.theta) * math.cos(self.phi)
            - math.cos(self.theta) * math.sin(self.phi) * math.sin(self.inclination)
        )
        self.y = self.r * (
            math.sin(self.theta) * math.sin(self.phi)
            + math.cos(self.theta) * math.cos(self.phi) * math.sin(self.inclination)
        )
        self.z = self.r * math.cos(self.theta) * math.cos(self.inclination)
        self.polar_angle = self.theta
        self.latitude = math.asin(self.z / self.r)

        if self.x > 0:
            self.longitude = math.atan(self.y / self.x)
        elif self.x < 0 and self.y >= 0:
            self.longitude = math.pi + math.atan(self.y / self.x)
        elif self.x < 0 and self.y < 0:
            self.longitude = math.atan(self.y / self.x) - math.pi
        elif self.y > 0:
            self.longitude = math.pi / 2
        elif self.y < 0:
            self.longitude = -math.pi / 2
        else:
            self.longitude = 0

    def getLinkLatencies(self, graph):
        latencies = []

        for buffer in self.sendBufferSatsIntra:
            dataRate = nx.path_weight(graph, [self.ID, buffer[2]], "dataRateOG")
            distance = nx.path_weight(graph, [self.ID, buffer[2]], "slant_range")
            bufferSize = len(buffer[1]) + 1

            propTime = distance / Vc
            transmitTime = (bufferSize * BLOCK_SIZE) / dataRate
            latency = propTime + transmitTime
            latencies.append([buffer[2], latency])

        for buffer in self.sendBufferSatsInter:
            dataRate = nx.path_weight(graph, [self.ID, buffer[2]], "dataRateOG")
            distance = nx.path_weight(graph, [self.ID, buffer[2]], "slant_range")
            bufferSize = len(buffer[1]) + 1

            propTime = distance / Vc
            transmitTime = (bufferSize * BLOCK_SIZE) / dataRate
            latency = propTime + transmitTime
            latencies.append([buffer[2], latency])

        return latencies


class DataBlock:
    """
    Datablock : 시뮬레이터 안에서 흐르는 '패킷' 단위
    - source와 destination 저장
    - 현재 path 저장
    - 각종 latency 저장 (queue, tx, prop, total)
    - hop별 도착/송신 시각 저장
    
    => 하나의 block 객체에서
        어디서 생성됐고, 어디로 가고,어떤 path를 이용했고,
        얼마나 기다렸고, 얼마나 전송됐고, 얼마나 전파됐는지 
    
    getTotalTransmissionTime() => 전체 전송 시간 계산
    (생성 이후 목적지에 도착할 때까지의 총 시간)

    getQueueTime() => 큐에서 기다린 시간
    * Queueing Delay = 처음 생성 후 송신까지 기다린 시간 + 각 hop에서 도착 후 송신까지 걸린 시간
    """
    def __init__(self, source, destination, ID, creationTime):
        self.size = BLOCK_SIZE
        self.destination = destination
        self.source = source
        self.ID = ID
        self.timeAtFull = None
        self.creationTime = creationTime
        self.timeAtFirstTransmission = None
        self.checkPoints = []
        self.checkPointsSend = []
        self.path = []
        self.queueLatency = (None, None)
        self.txLatency = 0
        self.propLatency = 0
        self.totLatency = 0
        self.isNewPath = False
        self.oldPath = []
        self.newPath = []

    def getQueueTime(self):
        queueLatency = [0, []]
        queueLatency[0] += self.timeAtFirstTransmission - self.creationTime
        queueLatency[1].append(self.timeAtFirstTransmission - self.creationTime)

        for arrived, sendReady in zip(self.checkPoints, self.checkPointsSend):
            queueLatency[0] += sendReady - arrived
            queueLatency[1].append(sendReady - arrived)

        self.queueLatency = queueLatency
        return queueLatency

    def getTotalTransmissionTime(self):
        totalTime = 0
        if len(self.checkPoints) == 1:
            return self.checkPoints[0] - self.timeAtFirstTransmission

        lastTime = self.creationTime
        for t in self.checkPoints:
            totalTime += t - lastTime
            lastTime = t

        self.totLatency = totalTime
        return totalTime

    def __repr__(self):
        return (
            f"ID = {self.ID}\n Source:\n {self.source}\n Destination:\n {self.destination}\n"
            f"Total latency: {self.totLatency}"
        )


class Gateway:
    """
    Gateway : 지상 GT (트래픽 생성하고 위성으로 올려보내는 역할)
    - 인구 cell (현재 dataset이 'Population Map')과 연결
    - 총 트래픽 양 계산
    - 목적지별 block 생성
    - uplink 전송
    - 목적지별 shortest path 저장
    ** 현재 GT는 1) 트래픽 발생 + 2) 송신 노드 + 3) 목적지별 경로 캐시

    ※ timeToFullBlock()에서 method
        : GT가 생성하는 총 트래픽을 각 목적지 GT에 어떻게 나눌지 정하는 방식
        1) fraction : 미리 계산된 fractions 테이블 이용 
            - 가장 유연, 수신측 downlink capacity와 같은 제약 반영 가능
            - src-dst 쌍마다 서로 다른 분배 가능
        2) CurrentNumb : 현재 활성화된 Gateway 수를 기준으로 균등 분배
            - 활성 GT가 많아지면 각 목적지 몫은 줄어듦
        3) totalNumb : 전체 가능한 GT 수를 기준으로 균등 분배
            - 현재 몇 개가 활성화되어 있는지 무시
            - 항상 같은 기준으로 나누기 때문에 실험 비교에 적합
        현재는 gt.makeFillBlockProcesses(self.gateways, "totalNumb", gtIndex) 처럼
        'totalNumb' 방식 사용

    ※ timeToFullBlock()에서 fractionIndex
        : src->dst 트래픽 비율을 찾기 위한 index
        : [source gateway index, destination gateway index]

    makeFillBlockProcesses() => 각 dest. GT에 대해 block 생성 process 만듦
    : source GT 하나당 #dest만큼 프로세스 발생

    fillBlock(destination, method, fractionIndex) => 실제로 blcok 생성 (=트래픽 생성기)
    * Workflow
    ** 1. block 생성
    ** 2. block이 다 찰 때까지 대기 (method와 관련)
    ** 3. path를 붙임
    ** 4. sendBuffer에 넣음 

    sendBlock() => GT가 linked satellite로 블록을 올려보내는 process
    * Workflow (uplink 송신기)
    ** 1. 큐에서 블록 꺼냄
    ** 2. uplink transmission delay 계산
    ** 3. 위성에 receive process 생성 

    receivedBlock()=> 목적지 GT에서 최종 수신할 때 호출
    ** receivedDataBlocks에 들어가서 실험 result 집계 대상

    adjustDataRate()=> GT->위성 uplink rate 계산

    getTotalFlow() => GT 서비스 영역 내 사용자 수를 기반으로 평균 총 트래픽 계산
        - 인구 data 기반으로 얼마나 많은 block이 생성될 지 정하는 함수

    updateGraph(block) => path metric이 'latency'일 때 현재 queue 상태를 반영 그래프 가중치 갱신
    ** latency metric은 static shortest path가 아니라 current queue-aware shortest path
    """
    def __init__(self, name, ID, latitude, longitude, totalX, totalY, totalGTs, env, totalLocations, earth, pathMetric):
        self.earth = earth

        self.name = name
        self.ID = ID
        self.latitude = latitude
        self.longitude = longitude

        self.gridLocationX = int((0.5 + longitude / 360) * totalX)
        self.gridLocationY = int((0.5 - latitude / 180) * totalY)
        self.cellsInRange = []
        self.totalGTs = totalGTs
        self.totalLocations = totalLocations
        self.totalAvgFlow = None
        self.totalX = totalX
        self.totalY = totalY

        self.polar_angle = (math.pi / 2 - math.radians(self.latitude) + 2 * math.pi) % (2 * math.pi)
        self.x = Re * math.cos(math.radians(self.longitude)) * math.sin(self.polar_angle)
        self.y = Re * math.sin(math.radians(self.longitude)) * math.sin(self.polar_angle)
        self.z = Re * math.cos(self.polar_angle)

        self.satsOrdered = []
        self.satIndex = 0
        self.linkedSat = (None, None)
        self.graph = nx.Graph()

        self.env = env
        self.datBlocks = []
        self.fillBlocks = []
        self.sendBlocks = env.process(self.sendBlock())
        self.sendBuffer = ([env.event()], [])
        self.paths = {}
        self.pathMetric = pathMetric

        self.receiveFraction = 0

        self.dataRate = None
        self.gs2ngeo = RFlink(
            frequency=30e9,
            bandwidth=500e6,
            maxPtx=20,
            aDiameterTx=0.33,
            aDiameterRx=0.26,
            pointingLoss=0.3,
            noiseFigure=2,
            noiseTemperature=290,
            min_rate=10e3
        )

    def makeFillBlockProcesses(self, Receivers, method, fractionIndex):
        self.totalGTs = len(Receivers)
        for receiverIndex, Receiver in enumerate(Receivers):
            if Receiver != self:
                self.fillBlocks.append(
                    self.env.process(self.fillBlock(Receiver, method, [fractionIndex, receiverIndex]))
                )

    def fillBlock(self, destination, method, fractionIndex):
        index = 0
        unavailableDestinationBuffer = []

        while True:
            try:
                block = DataBlock(
                    self,
                    destination,
                    str(self.ID) + "_" + str(destination.ID) + "_" + str(index),
                    self.env.now
                )

                timeToFull, _ = self.timeToFullBlock(block, method, fractionIndex)
                yield self.env.timeout(timeToFull)

                if block.destination.linkedSat[0] is None:
                    unavailableDestinationBuffer.append(block)
                else:
                    while unavailableDestinationBuffer:
                        if not self.sendBuffer[0][0].triggered:
                            self.sendBuffer[0][0].succeed()
                            self.sendBuffer[1].append(unavailableDestinationBuffer[0])
                            unavailableDestinationBuffer.pop(0)
                        else:
                            newEvent = self.env.event().succeed()
                            self.sendBuffer[0].append(newEvent)
                            self.sendBuffer[1].append(unavailableDestinationBuffer[0])
                            unavailableDestinationBuffer.pop(0)

                    block.path = self.paths[destination.name]
                    if not block.path:
                        print(self.name, destination.name)
                        raise RuntimeError("Empty path for block")

                    block.timeAtFull = self.env.now
                    createdBlocks.append(block)

                    if not self.sendBuffer[0][0].triggered:
                        self.sendBuffer[0][0].succeed()
                        self.sendBuffer[1].append(block)
                    else:
                        newEvent = self.env.event().succeed()
                        self.sendBuffer[0].append(newEvent)
                        self.sendBuffer[1].append(block)

                    index += 1
            except simpy.Interrupt:
                break

    def sendBlock(self):
        while True:
            yield self.sendBuffer[0][0]

            while self.linkedSat[0] is None:
                yield self.env.timeout(0.1)

            propTime = self.timeToSend(self.linkedSat)
            timeToSend = BLOCK_SIZE / self.dataRate

            self.sendBuffer[1][0].timeAtFirstTransmission = self.env.now
            yield self.env.timeout(timeToSend)
            self.sendBuffer[1][0].txLatency += timeToSend

            if not self.sendBuffer[1][0].path:
                print(self.sendBuffer[1][0].source.name, self.sendBuffer[1][0].destination.name)
                raise RuntimeError("Block path missing before GT send")

            if self.pathMetric == "latency":
                self.updateGraph(self.sendBuffer[1][0])

            self.linkedSat[1].createReceiveBlockProcess(self.sendBuffer[1][0], propTime)

            if len(self.sendBuffer[0]) == 1:
                self.sendBuffer[0].pop(0)
                self.sendBuffer[1].pop(0)
                self.sendBuffer[0].append(self.env.event())
            else:
                self.sendBuffer[0].pop(0)
                self.sendBuffer[1].pop(0)

    def timeToSend(self, linkedSat):
        return linkedSat[0] / Vc

    def createReceiveBlockProcess(self, block, propTime):
        self.env.process(self.receiveBlock(block, propTime))

    def receiveBlock(self, block, propTime):
        yield self.env.timeout(propTime)
        block.propLatency += propTime
        block.checkPoints.append(self.env.now)
        receivedDataBlocks.append(block)

    def cellDistance(self, cell):
        cellCoord = (math.degrees(cell.latitude), math.degrees(cell.longitude))
        gTCoord = (self.latitude, self.longitude)
        return geopy.distance.geodesic(cellCoord, gTCoord).km

    def distance_GSL(self, satellite):
        satCoords = [satellite.x, satellite.y, satellite.z]
        GTCoords = [self.x, self.y, self.z]
        return math.dist(satCoords, GTCoords)

    def adjustDataRate(self):
        speff_thresholds = np.array(
            [0, 0.434841, 0.490243, 0.567805, 0.656448, 0.789412, 0.889135, 0.988858,
             1.088581, 1.188304, 1.322253, 1.487473, 1.587196, 1.647211, 1.713601,
             1.779991, 1.972253, 2.10485, 2.193247, 2.370043, 2.458441, 2.524739,
             2.635236, 2.637201, 2.745734, 2.856231, 2.966728, 3.077225, 3.165623,
             3.289502, 3.300184, 3.510192, 3.620536, 3.703295, 3.841226, 3.951571,
             4.206428, 4.338659, 4.603122, 4.735354, 4.933701, 5.06569, 5.241514,
             5.417338, 5.593162, 5.768987, 5.900855]
        )
        lin_thresholds = np.array(
            [1e-10, 0.5188000389, 0.5821032178, 0.6266138647, 0.751622894, 0.9332543008,
             1.051961874, 1.258925412, 1.396368361, 1.671090614, 2.041737945, 2.529297996,
             2.937649652, 2.971666032, 3.25836701, 3.548133892, 3.953666201, 4.518559444,
             4.83058802, 5.508076964, 6.45654229, 6.886522963, 6.966265141, 7.888601176,
             8.452788452, 9.354056741, 10.49542429, 11.61448614, 12.67651866, 12.88249552,
             14.48771854, 14.96235656, 16.48162392, 18.74994508, 20.18366364, 23.1206479,
             25.00345362, 30.26913428, 35.2370871, 38.63669771, 45.18559444, 49.88844875,
             52.96634439, 64.5654229, 72.27698036, 76.55966069, 90.57326009]
        )

        pathLoss = 10 * np.log10((4 * math.pi * self.linkedSat[0] * self.gs2ngeo.f / Vc) ** 2)
        snr = 10 ** ((self.gs2ngeo.maxPtx_db + self.gs2ngeo.G - pathLoss - self.gs2ngeo.No) / 10)

        feasible_speffs = speff_thresholds[np.nonzero(lin_thresholds <= snr)]
        speff = self.gs2ngeo.B * feasible_speffs[-1]
        self.dataRate = speff

    def orderSatsByDist(self, constellation):
        sats = []
        index = 0
        for orbitalPlane in constellation:
            for sat in orbitalPlane.sats:
                d_GSL = self.distance_GSL(sat)
                if d_GSL <= sat.maxSlantRange():
                    sats.append((d_GSL, sat, [index]))
                index += 1
        sats.sort()
        self.satsOrdered = sats

    def addRefOnSat(self):
        if self.satIndex >= len(self.satsOrdered):
            self.linkedSat = (None, None)
            print(f"No satellite for GT {self.name}")
            return

        if self.satsOrdered[self.satIndex][1].linkedGT is None:
            self.satsOrdered[self.satIndex][1].linkedGT = self
            self.satsOrdered[self.satIndex][1].GTDist = self.satsOrdered[self.satIndex][0]

        elif self.satsOrdered[self.satIndex][1].GTDist < self.satsOrdered[self.satIndex][0]:
            self.satsOrdered[self.satIndex][1].linkedGT.satIndex += 1
            self.satsOrdered[self.satIndex][1].linkedGT.addRefOnSat()

            self.satsOrdered[self.satIndex][1].linkedGT = self
            self.satsOrdered[self.satIndex][1].GTDist = self.satsOrdered[self.satIndex][0]
        else:
            self.satIndex += 1
            if self.satIndex == len(self.satsOrdered):
                self.linkedSat = (None, None)
                print(f"No satellite for GT {self.name}")
                return
            self.addRefOnSat()

    def link2Sat(self, dist, sat):
        self.linkedSat = (dist, sat)
        sat.linkedGT = self
        sat.GTDist = dist
        self.adjustDataRate()

    def addCell(self, cellInfo):
        self.cellsInRange.append(cellInfo)

    def removeCell(self, cell):
        for i, cellInfo in enumerate(self.cellsInRange):
            if cell.latitude == cellInfo[0][0] and cell.longitude == cellInfo[0][1]:
                cellInfo.pop(i)
                return True
        return False

    def findCellsWithinRange(self, earth, maxDistance):
        isWithinRangeX = True
        x = self.gridLocationX
        while isWithinRangeX:
            y = self.gridLocationY
            isWithinRangeY = True
            if x == earth.total_x:
                x = 0
            cell = earth.cells[x][y]
            distance = self.cellDistance(cell)
            if distance > maxDistance:
                isWithinRangeY = False
                isWithinRangeX = False
            while isWithinRangeY:
                if y == -1:
                    y = earth.total_y - 1
                cell = earth.cells[x][y]
                distance = self.cellDistance(cell)
                if distance > maxDistance:
                    isWithinRangeY = False
                else:
                    if cell.gateway is None or (cell.gateway is not None and distance < cell.gateway[1]):
                        cell.gateway = (self, distance)
                y -= 1
            x += 1

        isWithinRangeX = True
        x = self.gridLocationX
        while isWithinRangeX:
            y = self.gridLocationY + 1
            isWithinRangeY = True
            if x == earth.total_x:
                x = 0
            cell = earth.cells[x][y]
            distance = self.cellDistance(cell)
            if distance > maxDistance:
                isWithinRangeY = False
                isWithinRangeX = False
            while isWithinRangeY:
                if y == earth.total_y:
                    y = 0
                cell = earth.cells[x][y]
                distance = self.cellDistance(cell)
                if distance > maxDistance:
                    isWithinRangeY = False
                else:
                    if cell.gateway is None or (cell.gateway is not None and distance < cell.gateway[1]):
                        cell.gateway = (self, distance)
                y += 1
            x += 1

        isWithinRangeX = True
        x = self.gridLocationX - 1
        while isWithinRangeX:
            y = self.gridLocationY
            isWithinRangeY = True
            if x == -1:
                x = earth.total_x - 1
            cell = earth.cells[x][y]
            distance = self.cellDistance(cell)
            if distance > maxDistance:
                isWithinRangeY = False
                isWithinRangeX = False
            while isWithinRangeY:
                if y == -1:
                    y = earth.total_y - 1
                cell = earth.cells[x][y]
                distance = self.cellDistance(cell)
                if distance > maxDistance:
                    isWithinRangeY = False
                else:
                    if cell.gateway is None or (cell.gateway is not None and distance < cell.gateway[1]):
                        cell.gateway = (self, distance)
                y -= 1
            x -= 1

        isWithinRangeX = True
        x = self.gridLocationX - 1
        while isWithinRangeX:
            y = self.gridLocationY + 1
            isWithinRangeY = True
            if x == -1:
                x = earth.total_x - 1
            cell = earth.cells[x][y]
            distance = self.cellDistance(cell)
            if distance > maxDistance:
                isWithinRangeY = False
                isWithinRangeX = False
            while isWithinRangeY:
                if y == -1:
                    y = earth.total_y - 1
                cell = earth.cells[x][y]
                distance = self.cellDistance(cell)
                if distance > maxDistance:
                    isWithinRangeY = False
                else:
                    if cell.gateway is None or (cell.gateway is not None and distance < cell.gateway[1]):
                        cell.gateway = (self, distance)
                y += 1
            x -= 1

    def timeToFullBlock(self, block, method, fractionIndex):
        if method == "fraction":
            flow = self.totalAvgFlow * self.earth.fractions[fractionIndex[0], fractionIndex[1]]
        elif method == "CurrentNumb":
            flow = self.totalAvgFlow / (self.totalGTs - 1)
        elif method == "totalNumb":
            flow = self.totalAvgFlow / (len(self.totalLocations) - 1)
        else:
            raise ValueError("incorrect method called")

        avgTime = block.size / flow
        time_ = np.random.exponential(scale=avgTime)
        return time_, flow

    def getTotalFlow(self, avgFlowPerUser, distanceFunc, maxDistance, capacity=None, fraction=1.0):
        totalAvgFlow = 0
        avgFlowPerUser = 8593 * 8

        if distanceFunc == "Step":
            for cell in self.cellsInRange:
                totalAvgFlow += cell[1] * avgFlowPerUser
        elif distanceFunc == "Slope":
            gradient = (0 - avgFlowPerUser) / (maxDistance - 0)
            for cell in self.cellsInRange:
                totalAvgFlow += (gradient * cell[2] + avgFlowPerUser) * cell[1]
        else:
            raise ValueError("distance function not recognized")

        if self.linkedSat[0] is None:
            self.dataRate = self.gs2ngeo.min_rate

        if not capacity:
            capacity = self.dataRate

        if totalAvgFlow < capacity * fraction:
            self.totalAvgFlow = totalAvgFlow
        else:
            self.totalAvgFlow = capacity * fraction

    def updateGraph(self, block):
        for plane in self.earth.LEO:
            for satellite in plane.sats:
                latencies = satellite.getLinkLatencies(self.graph)
                for latency in latencies:
                    nx.set_edge_attributes(self.graph, {(satellite.ID, latency[0]): {'latency': latency[1]}})
        block.path = getShortestPath(self.name, block.destination.name, "latency", self.graph)

    def __eq__(self, other):
        return self.latitude == other.latitude and self.longitude == other.longitude

    def __repr__(self):
        return (
            f"Location = {self.name}\n Longitude = {self.longitude}\n Latitude = {self.latitude}\n"
            f" pos x= {self.x}, pos y= {self.y}, pos z= {self.z}"
        )


class Cell:
    """
    Cell : 인구맵의 한 칸
     - 위도/경도 보유
     - 사용자 수(population 보유)
     - 어느 게이트웨이에 속할지 결정
     ** Cell은 트래픽 수요의 최소 단위

     setGT() : 가장 가까운 gateway를 찾고, 일정 거리 안에 있으면 그 gateway에 cell 정보 추가
    """
    def __init__(self, total_x, total_y, cell_x, cell_y, users, Re=6378e3, f=20e9, bw=200e6, noise_power=1 / (1e11)):
        self.map_x = cell_x
        self.map_y = cell_y
        self.latitude = math.pi * (0.5 - cell_y / total_y)
        self.longitude = (cell_x / total_x - 0.5) * 2 * math.pi

        self.area = 4 * math.pi * Re * Re * math.cos(self.latitude) / (total_x * total_y)
        self.x = Re * math.cos(self.latitude) * math.cos(self.longitude)
        self.y = Re * math.cos(self.latitude) * math.sin(self.longitude)
        self.z = Re * math.sin(self.latitude)

        self.users = users
        self.f = f
        self.bw = bw
        self.noise_power = noise_power
        self.rejected = True
        self.gateway = None

    def __repr__(self):
        return (
            f"Users = {self.users}\n area = {self.area / 1e6:.2f} km^2\n"
            f" longitude = {math.degrees(self.longitude):.2f} deg\n"
            f" latitude = {math.degrees(self.latitude):.2f} deg\n"
            f" pos x = {self.x:.2f}\n pos y = {self.y:.2f}\n pos z = {self.z:.2f}\n"
            f" x position on map = {self.map_x:.2f}\n y position on map = {self.map_y:.2f}"
        )

    def setGT(self, gateways, maxDistance=60):
        closestGT = (gateways[0], gateways[0].cellDistance(self))
        for gateway in gateways[1:]:
            distanceToGT = gateway.cellDistance(self)
            if distanceToGT < closestGT[1]:
                closestGT = (gateway, distanceToGT)
        self.gateway = closestGT

        if closestGT[1] <= maxDistance:
            closestGT[0].addCell(
                [(math.degrees(self.latitude), math.degrees(self.longitude)), self.users, closestGT[1]]
            )
        else:
            self.users = 0
        return closestGT


class Earth:
    """
    Earth : 전체 시뮬레이터의 최상위 환경 객체 (= 네트워크 전체를 담는 월드 모델)
    - 인구 맵 로드
    - cell grid 생성
    - gateway 생성
    - constellation 생성
    - GT<->cell 연결
    - GT<->sat 연결
    - constellation movement process 실행
    - GT path 갱신
    
    linkedSats2GTs(method) => 각 GT를 하나의 위성과 매칭 (Greedy 또는 Optimize 방식)
    : GSL 연결 설정

    updatedGTPaths() => 모든 GT->GT shortest path를 다시 계산
    ** constellation movement가 있거나 graph가 바뀌었을 때 사용

    getGSLDataRates(), getISLDataRates() : 링크 rate를 수집해 실험 결과로 저장할 때

    moveConstellationProcess() => Simpy Process로서 일정 시간마다 위성 위치를 이동시키고 topology 갱신
    (현재 코드에서는 거의 안 움직임)

    plotMap(), plot3D() => 시각화용 함수

    """
    def __init__(self, env, img_path, gt_path, constellation, inputParams, deltaT, totalLocations, getRates=False, window=None):
        pop_count_data = Image.open(img_path)

        pop_count = np.array(pop_count_data)
        pop_count[pop_count < 0] = 0

        [self.total_x, self.total_y] = pop_count_data.size
        self.total_cells = self.total_x * self.total_y

        self.cells = []
        for i in range(self.total_x):
            self.cells.append([])
            for j in range(self.total_y):
                self.cells[i].append(Cell(self.total_x, self.total_y, i, j, pop_count[j][i]))

        if window is not None:
            self.lati = [window[2], window[3]]
            self.longi = [window[0], window[1]]
            self.windowx = (
                int((0.5 + window[0] / 360) * self.total_x),
                int((0.5 + window[1] / 360) * self.total_x)
            )
            self.windowy = (
                int((0.5 - window[3] / 180) * self.total_y),
                int((0.5 - window[2] / 180) * self.total_y)
            )
        else:
            self.lati = [-90, 90]
            self.longi = [-179, 180]
            self.windowx = (0, self.total_x)
            self.windowy = (0, self.total_y)

        self.gateways = []
        gateways = pd.read_csv(gt_path)

        length = 0
        for i, location in enumerate(gateways['Location']):
            for name in inputParams['Locations']:
                if name in location.split(","):
                    length += 1

        if inputParams['Locations'][0] != 'All':
            for i, location in enumerate(gateways['Location']):
                for name in inputParams['Locations']:
                    if name in location.split(","):
                        lName = gateways['Location'][i]
                        gtLati = gateways['Latitude'][i]
                        gtLongi = gateways['Longitude'][i]
                        self.gateways.append(
                            Gateway(
                                lName, i, gtLati, gtLongi, self.total_x, self.total_y,
                                length, env, totalLocations, self, inputParams['Pathing'][0]
                            )
                        )
                        break
        else:
            for i in range(len(gateways['Latitude'])):
                name = gateways['Location'][i]
                gtLati = gateways['Latitude'][i]
                gtLongi = gateways['Longitude'][i]
                self.gateways.append(
                    Gateway(
                        name, i, gtLati, gtLongi, self.total_x, self.total_y,
                        len(gateways['Latitude']), env, totalLocations, self, inputParams['Pathing'][0]
                    )
                )

        if not getRates:
            for gtIndex, gt in enumerate(self.gateways):
                gt.makeFillBlockProcesses(self.gateways, "totalNumb", gtIndex)

        # local import으로 순환 import 회피
        from simulator import create_Constellation
        self.LEO = create_Constellation(constellation, env)

        self.pathParam = inputParams['Pathing'][0]
        self.moveConstellation = env.process(self.moveConstellationProcess(env, deltaT, getRates))

    def set_window(self, window):
        self.lati = [window[2], window[3]]
        self.longi = [window[0], window[1]]
        self.windowx = (
            int((0.5 + window[0] / 360) * self.total_x),
            int((0.5 + window[1] / 360) * self.total_x)
        )
        self.windowy = (
            int((0.5 - window[3] / 180) * self.total_y),
            int((0.5 - window[2] / 180) * self.total_y)
        )

    def linkCells2GTs(self, distance):
        start = time.time()
        for i, gt in enumerate(self.gateways):
            print(f"Finding cells within coverage area of GT {i+1} of {len(self.gateways)}", end='\r')
            gt.findCellsWithinRange(self, distance)
        print('\r')
        print(f"Time taken to find cells that are within range of all GTs: {time.time() - start} seconds")

        start = time.time()
        for cells in self.cells:
            for cell in cells:
                if cell.gateway is not None:
                    cell.gateway[0].addCell(
                        [(math.degrees(cell.latitude), math.degrees(cell.longitude)), cell.users, cell.gateway[1]]
                    )

        print(f"Time taken to add cell information to all GTs: {time.time() - start} seconds")
        print()

    def linkSats2GTs(self, method):
        sats = []
        for orbit in self.LEO:
            for sat in orbit.sats:
                sat.linkedGT = None
                sat.GTDist = None
                sats.append(sat)

        if method == "Greedy":
            for GT in self.gateways:
                GT.orderSatsByDist(self.LEO)
                GT.addRefOnSat()

            for orbit in self.LEO:
                for sat in orbit.sats:
                    if sat.linkedGT is not None:
                        sat.linkedGT.link2Sat(sat.GTDist, sat)

        elif method == "Optimize":
            SxGT = np.array([[99999 for _ in range(len(sats))] for _ in range(len(self.gateways))])

            for i, GT in enumerate(self.gateways):
                GT.orderSatsByDist(self.LEO)
                for val, entry in enumerate(GT.satsOrdered):
                    SxGT[i][entry[2][0]] = val

            rowInd, colInd = linear_sum_assignment(SxGT)

            for i, GT in enumerate(self.gateways):
                if SxGT[rowInd[i]][colInd[i]] < len(GT.satsOrdered):
                    sat = GT.satsOrdered[SxGT[rowInd[i]][colInd[i]]]
                    GT.link2Sat(sat[0], sat[1])
                else:
                    GT.linkedSat = (None, None)

    def getCellUsers(self):
        temp = []
        for i, cellList in enumerate(self.cells):
            temp.append([])
            for cell in cellList:
                temp[i].append(cell.users)
        return temp

    def updateGTPaths(self):
        for GT in self.gateways:
            for destination in self.gateways:
                if GT != destination:
                    if destination.linkedSat[0] is not None and GT.linkedSat[0] is not None:
                        path = getShortestPath(GT.name, destination.name, self.pathParam, GT.graph)
                        GT.paths.update({destination.name: path})
                    else:
                        GT.paths.update({destination.name: []})

            for block in GT.sendBuffer[1]:
                block.path = GT.paths[block.destination.name]
                block.isNewPath = True

    def getGSLDataRates(self):
        upDataRates = []
        downDataRates = []
        for GT in self.gateways:
            if GT.linkedSat[0] is not None:
                upDataRates.append(GT.dataRate)

        for orbit in self.LEO:
            for satellite in orbit.sats:
                if satellite.linkedGT is not None:
                    downDataRates.append(satellite.downRate)

        return upDataRates, downDataRates

    def getISLDataRates(self):
        interDataRates = []
        for orbit in self.LEO:
            for satellite in orbit.sats:
                for satData in satellite.interSats:
                    interDataRates.append(satData[2])
        return interDataRates

    def moveConstellationProcess(self, env, deltaT=3600, getRates=False):
        print(f"[MOVE PROCESS START] deltaT={deltaT}")
        if getRates and self.LEO and self.LEO[0].sats and self.LEO[0].sats[0].intraSats:
            intraRate.append(self.LEO[0].sats[0].intraSats[0][2])

        while True:
            if getRates:
                upDataRates, downDataRates = self.getGSLDataRates()
                inter = self.getISLDataRates()

                for val in upDataRates:
                    upGSLRates.append(val)
                for val in downDataRates:
                    downGSLRates.append(val)
                for val in inter:
                    interRates.append(val)

            yield env.timeout(deltaT)
            print(f"[MOVE TRIGGERED] env.now={env.now}")

            before_sat = self.LEO[0].sats[0]
            print(f"[BEFORE ROTATE] sat={before_sat.ID}, "
                  f"lat={math.degrees(before_sat.latitude):.6f}, "
                  f"lon={math.degrees(before_sat.longitude):.6f}"
                  )

            for GT in self.gateways:
                GT.satsOrdered = []
                GT.linkedSat = (None, None)

            for constellation in self.LEO:
                constellation.rotate(deltaT)
            
            after_sat = self.LEO[0].sats[0]
            print(
                f"[AFTER ROTATE] sat={after_sat.ID}, "
                f"lat = {math.degrees(after_sat.latitude):6f}, "
                f"lon = {math.degrees(after_sat.longitude):6f}"
            )

            self.linkSats2GTs("Optimize")

            graph = createGraph(self)
            self.graph = graph
            for GT in self.gateways:
                GT.graph = graph

            # 원본의 복잡한 satellite buffer 재배치 로직은 1차 분리 단계에선 생략
            # movementTime이 크게 설정되면 실험 중 호출되지 않음
            self.updateGTPaths()
    def plotMap(self, plotGT=True, plotSat=True, path=None, bottleneck=None):
        if plotGT:
            scat1 = None
            for GT in self.gateways:
                scat1 = plt.scatter(
                    GT.gridLocationX,
                    GT.gridLocationY,
                    marker='x',
                    c='red',
                    s=90,
                    linewidth=2
                )

        colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(self.LEO)))

        if plotSat:
            scat2 = None
            for plane, c in zip(self.LEO, colors):
                for sat in plane.sats:
                    gridSatX = int((0.5 + math.degrees(sat.longitude) / 360) * 1440)
                    gridSatY = int((0.5 - math.degrees(sat.latitude) / 180) * 720)
                    scat2 = plt.scatter(
                        gridSatX,
                        gridSatY,
                        marker='o',
                        s=45,
                        linewidth=0.5,
                        color=c
                    )

        if path:
            xValues = []
            yValues = []
            for hop in path:
                xValues.append(int((0.5 + hop[1] / 360) * 1440))
                yValues.append(int((0.5 - hop[2] / 180) * 720))
            plt.plot(
                xValues,
                yValues,
                color='yellow',
                linewidth=2.5
            )

        if plotSat and plotGT and scat1 is not None and scat2 is not None:
            plt.legend([scat1, scat2], ['Concentrators', 'Satellites'], loc=3, prop={'size': 8})
        elif plotSat and scat2 is not None:
            plt.legend([scat2], ['Satellites'], loc=3, prop={'size': 8})
        elif plotGT and scat1 is not None:
            plt.legend([scat1], ['Concentrators'], loc=3, prop={'size': 8})

        plt.xticks([])
        plt.yticks([])

        plt.imshow(
            np.log10(np.array(self.getCellUsers()).transpose() + 1), cmap = 'viridis'
        )

    def plot3D(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        xs, ys, zs = [], [], []
        xG, yG, zG = [], [], []

        for con in self.LEO:
            for sat in con.sats:
                xs.append(sat.x)
                ys.append(sat.y)
                zs.append(sat.z)
        ax.scatter(xs, ys, zs, marker='o')

        for GT in self.gateways:
            xG.append(GT.x)
            yG.append(GT.y)
            zG.append(GT.z)
        ax.scatter(xG, yG, zG, marker='^')
        plt.show()

    def __repr__(self):
        return (
            f"total divisions in x = {self.total_x}\n total divisions in y = {self.total_y}\n"
            f" total cells = {self.total_cells}\n window of operation (longitudes) = {self.windowx}\n"
            f" window of operation (latitudes) = {self.windowy}"
        )