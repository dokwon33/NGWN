import os
import time
import simpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from simulator import initialize
from sim_state import receivedDataBlocks, createdBlocks, upGSLRates, downGSLRates, interRates
from stats import getBlockTransmissionStats, simProgress
from plotting import plotLatencies
from models import BlocksForPickle
from live_view import live_plot_simulation

def find_gateway(gateways, name):
    for i, g in enumerate(gateways):
        if name in g.name:
            return i
    return None


def main():
    """
    main.py 역할 : input.csv 읽기 / 실험 루프 실행 / 결과 저장 호출
    """
    percentages = {
        'Queue time': [],
        'Propagation time': [],
        'Transmission time': [],
        'GTnumber': []
    }

    inputParams = pd.read_csv("input.csv")
    ## locations = inputParams['Locations'].copy()
    locations = pd.Series(["Seoul", "Boston"]) # 서울과 보스턴 => 활성화 게이트웨이 목록 무시

    pathing = inputParams['Pathing'][0]
    testType = inputParams['Test type'][0]
    testLength = inputParams['Test length'][0]

    movementTime = 2 # 실험 루프용으로 10*3600으로 복구 필요
    ## movenmentTime과 dt에 대한 이해가 필요하며 적절한 설정이 요구

    savePath1 = f"./Results/latency Test/{pathing} {int(testLength)}s/"
    os.makedirs(savePath1, exist_ok=True)

    if testType == "Rates":
        numberOfMovements = testLength
        simulationTimelimit = movementTime * numberOfMovements + 10
    else:
        simulationTimelimit = max(testLength, 100) # 실험 루프용으로 simulationTimelimit = testlength으로 복구

    savePath2 = savePath1 + "{}/".format(testType)
    os.makedirs(savePath2, exist_ok=True)

    blockPath = "./Results/Congestion_test/{} {}s/".format(pathing, int(testLength))
    os.makedirs(blockPath, exist_ok=True)

    for GTnumber in range(2, 3): # 실험 루프 용으로 range(2, 19)로 복구 필요
        env = simpy.Environment()

        inputParams['Locations'] = locations[:GTnumber]

        earth1, graph1, bottleneck1, bottleneck2 = initialize(
            env,
            'Population Map/gpw_v4_population_count_rev11_2020_15_min.tif',
            'Gateways.csv',
            500,
            inputParams,
            movementTime,
            locations
        )
        src = find_gateway(earth1.gateways, "Seoul")
        dst = find_gateway(earth1.gateways, "Boston")
        print("before live_plot, env.now =", env.now)
        live_plot_simulation(env, earth1, simulationTimelimit, dt=2, path_pair=(src, dst))
        print("after live_plot, env.now =", env.now)

        # continue 제거 (3/17)
        ## (3/17 lab) 검증 및 평가에 사용할 수 있을 때 csv파일로 뽑기!
        ### 추후에 최적의 path이다 ! 라고 말하려면 검증이 필요하기 때문
        env.run(simulationTimelimit) ## -> 시각적 요소 제외

        progress = env.process(simProgress(simulationTimelimit, env))
        startTime = time.time()
        env.run(simulationTimelimit)
        timeToSim = time.time() - startTime

        if testType == "Rates":
            ratesPath = "./Results/Rates Test/"
            os.makedirs(ratesPath, exist_ok=True)

            data = {"upGSLRates": upGSLRates}
            d = pd.DataFrame(data=data)
            d.to_csv(ratesPath + "UpLinkRates.csv", index=False)

            data = {"downGSLRates": downGSLRates}
            d = pd.DataFrame(data=data)
            d.to_csv(ratesPath + "DownLinkRates.csv", index=False)

            data = {"interRates": interRates}
            d = pd.DataFrame(data=data)
            d.to_csv(ratesPath + "InterLinkRates.csv", index=False)

            plt.clf()
            plt.hist(np.asarray(interRates) / 1e9, cumulative=1, histtype='step', density=True)
            plt.title('CDF - Inter plane ISL data rates')
            plt.ylabel('Empirical CDF')
            plt.xlabel('Data rate [Gbps]')
            plt.savefig(ratesPath + "InterRatesCDF.png")

            plt.clf()
            plt.hist(np.asarray(upGSLRates) / 1e9, cumulative=1, histtype='step', density=True)
            plt.title('CDF - Uplink data rates')
            plt.ylabel('Empirical CDF')
            plt.xlabel('Data rate [Gbps]')
            plt.savefig(ratesPath + "UpLinkRatesCDF.png")

            plt.clf()
            plt.hist(np.asarray(downGSLRates) / 1e9, cumulative=1, histtype='step', density=True)
            plt.title('CDF - Downlink data rates')
            plt.ylabel('Empirical CDF')
            plt.xlabel('Data rate [Gbps]')
            plt.savefig(ratesPath + "DownLinkRatesCDF.png")
            plt.clf()

        else:
            results = getBlockTransmissionStats(
                timeToSim,
                inputParams['Locations'],
                inputParams['Constellation'][0]
            )

            pathBlocks = [[], []]

            first = earth1.gateways[0]
            second = earth1.gateways[1]
            allLatencies = []

            for block in receivedDataBlocks:
                allLatencies.append(block.totLatency)
                if block.source == first and block.destination == second:
                    pathBlocks[0].append(block.totLatency)
                    pathBlocks[1].append(block)

            xs = [l for l in range(len(allLatencies))]
            plt.figure()
            plt.scatter(xs, allLatencies, c='b')
            plt.ylabel('Latency')

            data = {"latencies": pathBlocks[0]}
            d = pd.DataFrame(data=data)
            d.to_csv(savePath1 + "pathLatencies_{}gateways.csv".format(GTnumber), index=False)

            data = {"latencies": allLatencies}
            d = pd.DataFrame(data=data)
            d.to_csv(savePath1 + "allLatencies_{}gateways.csv".format(GTnumber), index=False)

            plt.savefig(savePath2 + '{}_gateways.png'.format(GTnumber))
            plt.clf()
            plt.close()

            percentages['Queue time'].append(results.meanQueueLatency)
            percentages['Propagation time'].append(results.meanPropLatency)
            percentages['Transmission time'].append(results.meanTransLatency)
            percentages['GTnumber'].append(GTnumber)

            longest = 0
            longBlock = None
            for block in receivedDataBlocks:
                longTime = block.getTotalTransmissionTime()
                if longTime > longest:
                    longest = longTime
                    longBlock = block

            if longBlock is not None:
                print(longBlock.checkPoints)
                print(longBlock.queueLatency)
                print(longest)
                print(longBlock.txLatency)
                print(longBlock.propLatency)
                print(longBlock.totLatency)

            blocks = []
            for block in receivedDataBlocks:
                blocks.append(BlocksForPickle(block))
            np.save("{}blocks_{}".format(blockPath, GTnumber), np.asarray(blocks), allow_pickle=True)

            receivedDataBlocks.clear()
            createdBlocks.clear()

    if testType != "Rates":
        plotLatencies(percentages, pathing, savePath2)


if __name__ == '__main__':
    main()