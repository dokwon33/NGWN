import math
from constants import Vc, k, eff

class Results:
    def __init__(self, finishedBlocks, constellation, GTs, meanTotalLatency,
                 meanQueueLatency, meanTransLatency, meanPropLatency,
                 perQueueLatency, perPropLatency, perTransLatency):
        """
        저장하는 값 : 평균 총 지연, 평균 큐 지연, 평균 전파 지연, 평균 전송 지연, 각 지연 비율
        """
        self.GTs = GTs
        self.finishedBlocks = finishedBlocks
        self.constellation = constellation
        self.meanTotalLatency = meanTotalLatency
        self.meanQueueLatency = meanQueueLatency
        self.meanPropLatency = meanPropLatency
        self.meanTransLatency = meanTransLatency
        self.perQueueLatency = perQueueLatency
        self.perPropLatency = perPropLatency
        self.perTransLatency = perTransLatency

class BlocksForPickle:
    def __init__(self, block): 
        """
        BlocksForPickle : 실험 루프 끝난 후 block 정보를 .npy로 저장하기 위한 클래스
        => Datablock 객체를 그대로 저장하지 않고, 필요한 정보만 추려서 저장
        """
        self.size = 64800
        self.ID = block.ID
        self.timeAtFull = block.timeAtFull
        self.creationTime = block.creationTime
        self.timeAtFirstTransmission = block.timeAtFirstTransmission
        self.checkPoints = block.checkPoints
        self.checkPointsSend = block.checkPointsSend
        self.path = block.path
        self.queueLatency = block.queueLatency
        self.txLatency = block.txLatency
        self.propLatency = block.propLatency
        self.totLatency = block.totLatency

class RFlink:
    def __init__(self, frequency, bandwidth, maxPtx, aDiameterTx, aDiameterRx,
                 pointingLoss, noiseFigure, noiseTemperature, min_rate):
        """
        RFlink : 주파수 / 대역폭 / 송신 전력 / 안테나 이득 / noise power / 최소 rate
        => GSL / ISL의 데이터율 계산용 링크 모델
        """
        self.f = frequency
        self.B = bandwidth
        self.maxPtx = maxPtx
        self.maxPtx_db = 10 * math.log10(self.maxPtx)
        self.Gtx = 10 * math.log10(eff * ((math.pi * aDiameterTx * self.f / Vc) ** 2))
        self.Grx = 10 * math.log10(eff * ((math.pi * aDiameterRx * self.f / Vc) ** 2))
        self.G = self.Gtx + self.Grx - 2 * pointingLoss
        self.No = 10 * math.log10(self.B * k) + noiseFigure + 10 * math.log10(
            290 + (noiseTemperature - 290) * (10 ** (-noiseFigure / 10)))
        self.GoT = 10 * math.log10(eff * ((math.pi * aDiameterRx * self.f / Vc) ** 2)) \
                   - noiseFigure - 10 * math.log10(
            290 + (noiseTemperature - 290) * (10 ** (-noiseFigure / 10)))
        self.min_rate = min_rate

class FSOlink:
    def __init__(self, data_rate, power, comm_range, weight):
        """
        FSOlink : 광통신 링크용 클래스 (현재 핵심적으로 사용 X)
        """
        self.data_rate = data_rate
        self.power = power
        self.comm_range = comm_range
        self.weight = weight