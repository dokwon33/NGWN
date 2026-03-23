# plotting.py
import pandas as pd
import matplotlib.pyplot as plt


def plotShortestPath(earth, path):
    earth.plotMap(True, True, path=path)
    plt.savefig('popMap_500.pdf', dpi=500)
    plt.show()


def plotLatencies(percentages, pathing, savePath):
    """
    stacked bar plot
    """
    barWidth = 0.85
    r = percentages['GTnumber']
    numbers = percentages['GTnumber']
    GTnumber = len(r)

    plt.figure()

    plt.bar(
        r,
        percentages['Propagation time'],
        color='#b5ffb9',
        edgecolor='white',
        width=barWidth,
        label="Propagation time"
    )
    plt.bar(
        r,
        percentages['Queue time'],
        bottom=percentages['Propagation time'],
        color='#f9bc86',
        edgecolor='white',
        width=barWidth,
        label="Queue time"
    )
    plt.bar(
        r,
        percentages['Transmission time'],
        bottom=[
            i + j for i, j in zip(percentages['Propagation time'], percentages['Queue time'])
        ],
        color='#a3acff',
        edgecolor='white',
        width=barWidth,
        label="Transmission time"
    )

    plt.xticks(numbers)
    plt.xlabel("Nº of gateways")
    plt.ylabel("Latency")
    plt.legend(loc='lower left')

    plt.savefig(savePath + '{}_gatewaysTotal.png'.format(GTnumber))

    data = {
        "numb gateways": r,
        "prop delay": percentages['Propagation time'],
        "Queue delay": percentages['Queue time'],
        "transmission delay": percentages['Transmission time']
    }
    d = pd.DataFrame(data=data)
    d.to_csv(savePath + "delayFractions.csv", index=False)

    plt.close()