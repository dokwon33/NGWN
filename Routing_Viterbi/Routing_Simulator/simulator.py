# simulator.py
import numpy as np

from entities import Earth, OrbitalPlane
from routing import createGraph, getShortestPath, findBottleneck
from sim_state import (
    receivedDataBlocks,
    createdBlocks,
    upGSLRates,
    downGSLRates,
    interRates,
    intraRate,
)


def create_Constellation(specific_constellation, env):
    """
    위성군 생성
    """
    if specific_constellation == "small":
        print("Using small walker Star constellation")
        P = 3
        N_p = 4
        height = 1000e3
        inclination_angle = 53
        Walker_star = True
        min_elevation_angle = 30

    elif specific_constellation == "Kepler":
        print("Using Kepler constellation design")
        P = 7
        N_p = 20
        height = 600e3
        inclination_angle = 98.6
        Walker_star = True
        min_elevation_angle = 30

    elif specific_constellation == "Iridium_NEXT":
        print("Using Iridium NEXT constellation design")
        P = 6
        N_p = 11
        height = 780e3
        inclination_angle = 86.4
        Walker_star = True
        min_elevation_angle = 30

    elif specific_constellation == "OneWeb":
        print("Using OneWeb constellation design")
        P = 18
        N = 648
        N_p = int(N / P)
        height = 1200e3
        inclination_angle = 86.4
        Walker_star = True
        min_elevation_angle = 30

    elif specific_constellation == "Starlink":
        print("Using Starlink constellation design")
        P = 72
        N = 1584
        N_p = int(N / P)
        height = 550e3
        inclination_angle = 53
        Walker_star = False
        min_elevation_angle = 25

    elif specific_constellation == "Test":
        print("Using a test constellation design")
        P = 30
        N = 1200
        N_p = int(N / P)
        height = 600e3
        inclination_angle = 86.4
        Walker_star = True
        min_elevation_angle = 30

    else:
        raise ValueError("Not valid Constellation Name")

    distribution_angle = 2 * np.pi
    if Walker_star:
        distribution_angle /= 2

    orbital_planes = []
    for i in range(0, P):
        orbital_planes.append(
            OrbitalPlane(
                str(i),
                height,
                i * distribution_angle / P,
                np.radians(inclination_angle),
                N_p,
                min_elevation_angle,
                str(i) + "_",
                env
            )
        )

    return orbital_planes


def initialize(env, popMapLocation, GTLocation, distance, inputParams, movementTime, totalLocations):
    """
    Earth / GT / Satellite / Graph / Path / Process 초기화
    """
    constellationType = inputParams['Constellation'][0]
    fraction = inputParams['Fraction'][0]
    testType = inputParams['Test type'][0]

    getRates = testType == "Rates"

    earth = Earth(
        env,
        popMapLocation,
        GTLocation,
        constellationType,
        inputParams,
        movementTime,
        totalLocations,
        getRates
    )

    print(earth)
    print()

    earth.linkCells2GTs(distance)
    earth.linkSats2GTs("Optimize")
    graph = createGraph(earth)

    for gt in earth.gateways:
        gt.graph = graph

    paths = []
    for GT in earth.gateways:
        for destination in earth.gateways:
            if GT != destination:
                if destination.linkedSat[0] is not None and GT.linkedSat[0] is not None:
                    path = getShortestPath(GT.name, destination.name, earth.pathParam, GT.graph)
                    GT.paths[destination.name] = path
                    paths.append(path)

    sats = []
    for plane in earth.LEO:
        for sat in plane.sats:
            sats.append(sat)

    for plane in earth.LEO:
        for sat in plane.sats:
            if sat.linkedGT is not None:
                sat.adjustDownRate()
                sat.sendBlocksGT.append(sat.env.process(sat.sendBlock((sat.GTDist, sat.linkedGT), False)))

            neighbors = list(graph.neighbors(sat.ID))
            itt = 0

            for sat2 in sats:
                if sat2.ID in neighbors:
                    dataRate = graph[sat2.ID][sat.ID]["dataRateOG"]
                    distance_ = graph[sat2.ID][sat.ID]["slant_range"]

                    if sat2.in_plane == sat.in_plane:
                        sat.intraSats.append((distance_, sat2, dataRate))
                        sat.sendBufferSatsIntra.append(([sat.env.event()], [], sat2.ID))
                        sat.sendBlocksSatsIntra.append(
                            sat.env.process(sat.sendBlock((distance_, sat2, dataRate), True, True))
                        )
                    else:
                        sat.interSats.append((distance_, sat2, dataRate))
                        sat.sendBufferSatsInter.append(([sat.env.event()], [], sat2.ID))
                        sat.sendBlocksSatsInter.append(
                            sat.env.process(sat.sendBlock((distance_, sat2, dataRate), True, False))
                        )

                    itt += 1
                    if itt == len(neighbors):
                        break

    bottleneck1, bottleneck2 = None, None
    if len(paths) >= 2:
        bottleneck2, minimum2 = findBottleneck(paths[1], earth, False)
        bottleneck1, minimum1 = findBottleneck(paths[0], earth, False, minimum2)

    for GT in earth.gateways:
        mins = []
        if GT.linkedSat[0] is not None:
            for pathKey in GT.paths:
                _, minimum = findBottleneck(GT.paths[pathKey], earth)
                mins.append(minimum)

            if GT.dataRate < GT.linkedSat[1].downRate:
                GT.getTotalFlow(1, "Step", 1, GT.dataRate, fraction)
            else:
                GT.getTotalFlow(1, "Step", 1, GT.linkedSat[1].downRate, fraction)

    return earth, graph, bottleneck1, bottleneck2


def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr)) * diff) / diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr