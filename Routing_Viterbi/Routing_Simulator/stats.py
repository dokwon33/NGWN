import time
import numpy as np
from models import Results

def getBlockTransmissionStats(timeToSim, GTs, constellationType, createdBlocks, receivedDataBlocks):
    allTransmissionTimes = []
    largestTransmissionTime = (0, None)
    mostHops = (0, None)
    queueLat = []
    txLat = []
    propLat = []
    blocks = []

    for block in receivedDataBlocks:
        time_ = block.getTotalTransmissionTime()
        hops = len(block.checkPoints)

        if largestTransmissionTime[0] < time_:
            largestTransmissionTime = (time_, block)

        if mostHops[0] < hops:
            mostHops = (hops, block)

        allTransmissionTimes.append(time_)
        queueLat.append(block.getQueueTime()[0])
        txLat.append(block.txLatency)
        propLat.append(block.propLatency)

    avgTime = np.mean(allTransmissionTimes)
    totalTime = sum(allTransmissionTimes)

    print("\n########## Results #########\n")
    print(f"The simulation took {timeToSim} seconds to run")
    print(f"A total of {len(createdBlocks)} data blocks were created")
    print(f"A total of {len(receivedDataBlocks)} data blocks were transmitted")
    print(f"A total of {len(createdBlocks) - len(receivedDataBlocks)} data blocks were stuck")
    print(f"Average transmission time for all blocks were {avgTime}")

    return Results(
        finishedBlocks=blocks,
        constellation=constellationType,
        GTs=GTs,
        meanTotalLatency=avgTime,
        meanQueueLatency=np.mean(queueLat),
        meanPropLatency=np.mean(propLat),
        meanTransLatency=np.mean(txLat),
        perQueueLatency=sum(queueLat)/totalTime*100,
        perPropLatency=sum(propLat)/totalTime*100,
        perTransLatency=sum(txLat)/totalTime*100
    )

def simProgress(simTimelimit, env):
    timeSteps = 100
    timeStepSize = simTimelimit / timeSteps
    progress = 1
    startTime = time.time()
    yield env.timeout(timeStepSize)
    while True:
        elapsedTime = time.time() - startTime
        estimatedTimeRemaining = elapsedTime * (timeSteps / progress) - elapsedTime
        print(
            f"Simulation progress: {progress}% Estimated time remaining: {int(estimatedTimeRemaining)} seconds Current simulation time: {env.now}",
            end='\r'
        )
        yield env.timeout(timeStepSize)
        progress += 1