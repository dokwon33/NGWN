# routing.py
import math
import numpy as np
import numba
import networkx as nx

from constants import Re, Vc
from models import RFlink


class edge:
    def __init__(self, sati, satj, slant_range, dji, dij, shannonRate):
        self.i = sati
        self.j = satj
        self.slant_range = slant_range
        self.dji = dji
        self.dij = dij
        self.shannonRate = shannonRate

    def __repr__(self):
        return (
            f"\n node i: {self.i}, node j: {self.j}, "
            f"slant_range: {self.slant_range}, shannonRate: {self.shannonRate}"
        )


def get_direction(Satellites):
    """
    각 위성이 다른 위성을 볼 때 안테나 방향성을 계산.
    """
    N = len(Satellites)
    direction = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        epsilon = -Satellites[i].inclination
        for j in range(N):
            direction[i, j] = np.sign(
                Satellites[i].y * math.sin(epsilon)
                + Satellites[i].z * math.cos(epsilon)
                - Satellites[j].y * math.sin(epsilon)
                - Satellites[j].z * math.cos(epsilon)
            )
    return direction


def get_pos_vectors_omni(Satellites):
    """
    위성들의 x,y,z 좌표와 궤도면 index 반환
    """
    N = len(Satellites)
    Positions = np.zeros((N, 3))
    meta = np.zeros(N, dtype=np.int_)
    for n in range(N):
        Positions[n, :] = [Satellites[n].x, Satellites[n].y, Satellites[n].z]
        meta[n] = Satellites[n].in_plane
    return Positions, meta


def get_slant_range(edge_obj):
    return edge_obj.slant_range


@numba.jit
def get_slant_range_optimized(Positions, N):
    """
    모든 위성 쌍 간 거리 행렬 계산
    """
    slant_range = np.zeros((N, N))
    for i in range(N):
        slant_range[i, i] = math.inf
        for j in range(i + 1, N):
            slant_range[i, j] = np.linalg.norm(Positions[i, :] - Positions[j, :])
    slant_range += np.transpose(slant_range)
    return slant_range


@numba.jit
def los_slant_range(_slant_range, _meta, _max, _Positions):
    """
    line-of-sight 가능한 링크만 남기고, 아니면 inf 처리
    """
    _slant_range_new = np.copy(_slant_range)
    _N = len(_slant_range)
    for i in range(_N):
        for j in range(_N):
            if _slant_range_new[i, j] > _max[_meta[i], _meta[j]]:
                _slant_range_new[i, j] = math.inf
    return _slant_range_new


def get_data_rate(_slant_range_los, interISL):
    """
    slant range 기반으로 Shannon rate 근사 계산
    """
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

    pathLoss = 10 * np.log10((4 * math.pi * _slant_range_los * interISL.f / Vc) ** 2)
    snr = 10 ** ((interISL.maxPtx_db + interISL.G - pathLoss - interISL.No) / 10)
    shannonRate = interISL.B * np.log2(1 + snr)

    speffs = np.zeros((len(_slant_range_los), len(_slant_range_los)))

    for n in range(len(_slant_range_los)):
        for m in range(len(_slant_range_los)):
            feasible_speffs = speff_thresholds[np.nonzero(lin_thresholds <= snr[n, m])]
            if feasible_speffs.size == 0:
                speffs[n, m] = 0
            else:
                speffs[n, m] = interISL.B * feasible_speffs[-1]

    return speffs


def markovianMatchingTwo(earth):
    """
    inter-plane / intra-plane 링크 생성
    """
    _A_Markovian = []
    Satellites = []
    W_M = []
    covered = set()

    for plane in earth.LEO:
        for sat in plane.sats:
            Satellites.append(sat)

    N = len(Satellites)

    interISL = RFlink(
        frequency=26e9,
        bandwidth=500e6,
        maxPtx=10,
        aDiameterTx=0.26,
        aDiameterRx=0.26,
        pointingLoss=0.3,
        noiseFigure=2,
        noiseTemperature=290,
        min_rate=10e3
    )

    M = len(earth.LEO)
    Max_slnt_rng = np.zeros((M, M))

    Orb_heights = []
    for plane in earth.LEO:
        Orb_heights.append(plane.h)

    for _i in range(M):
        for _j in range(M):
            Max_slnt_rng[_i, _j] = (
                np.sqrt((Orb_heights[_i] + Re) ** 2 - Re ** 2)
                + np.sqrt((Orb_heights[_j] + Re) ** 2 - Re ** 2)
            )

    direction = get_direction(Satellites)
    Positions, meta = get_pos_vectors_omni(Satellites)
    slant_range = get_slant_range_optimized(Positions, N)
    slant_range_los = los_slant_range(slant_range, meta, Max_slnt_rng, Positions)
    shannonRate = get_data_rate(slant_range_los, interISL)

    for i in range(N):
        for j in range(i + 1, N):
            if (
                Satellites[i].in_plane != Satellites[j].in_plane
                and ((i, direction[i, j]) not in covered)
                and ((j, direction[j, i]) not in covered)
            ):
                if slant_range_los[i, j] < 6000e3:
                    W_M.append(
                        edge(
                            Satellites[i].ID,
                            Satellites[j].ID,
                            slant_range_los[i, j],
                            direction[i, j],
                            direction[j, i],
                            shannonRate[i, j]
                        )
                    )

    W_sorted = sorted(W_M, key=get_slant_range)

    while W_sorted:
        candidate = W_sorted[0]
        if (
            (candidate.i, candidate.dji) not in covered
            and (candidate.j, candidate.dij) not in covered
        ):
            _A_Markovian.append(candidate)
            covered.add((candidate.i, candidate.dji))
            covered.add((candidate.j, candidate.dij))
        W_sorted.pop(0)

    nPlanes = len(earth.LEO)
    for plane in earth.LEO:
        nPerPlane = len(plane.sats)
        for sat in plane.sats:
            sat.findNeighbours(earth)

            i = sat.in_plane * nPerPlane + sat.i_in_plane

            j = sat.upper.in_plane * nPerPlane + sat.upper.i_in_plane
            _A_Markovian.append(
                edge(sat.ID, sat.upper.ID, slant_range_los[i, j], direction[i, j], direction[j, i], shannonRate[i, j])
            )

            j = sat.lower.in_plane * nPerPlane + sat.lower.i_in_plane
            _A_Markovian.append(
                edge(sat.ID, sat.lower.ID, slant_range_los[i, j], direction[i, j], direction[j, i], shannonRate[i, j])
            )

    return _A_Markovian


def createGraph(earth):
    """
    gateway + satellite graph 생성
    """
    g = nx.Graph()

    for plane in earth.LEO:
        for sat in plane.sats:
            g.add_node(sat.ID, sat=sat)

    for GT in earth.gateways:
        if GT.linkedSat[1]:
            g.add_node(GT.name, GT=GT)
            g.add_edge(
                GT.name,
                GT.linkedSat[1].ID,
                slant_range=GT.linkedSat[0],
                dataRate=1 / GT.dataRate,
                invDataRate=1 / GT.dataRate,
                dataRateOG=GT.dataRate,
                hop=1,
                latency=1
            )

    markovEdges = markovianMatchingTwo(earth)
    for markovEdge in markovEdges:
        g.add_edge(
            markovEdge.i,
            markovEdge.j,
            slant_range=markovEdge.slant_range,
            dataRate=1 / markovEdge.shannonRate,
            dataRateOG=markovEdge.shannonRate,
            hop=1,
            latency=1
        )

    return g


def getShortestPath(source, destination, weight, g):
    """
    networkx shortest_path wrapper
    """
    path = []
    try:
        shortest = nx.shortest_path(g, source, destination, weight=weight)
        for hop in shortest:
            key = list(g.nodes[hop])[0]
            if shortest.index(hop) == 0 or shortest.index(hop) == len(shortest) - 1:
                path.append([hop, g.nodes[hop][key].longitude, g.nodes[hop][key].latitude])
            else:
                path.append([
                    hop,
                    math.degrees(g.nodes[hop][key].longitude),
                    math.degrees(g.nodes[hop][key].latitude)
                ])
    except Exception:
        print(f'No path between {source} and {destination}, check graph.')
        return -1
    return path


def findBottleneck(path, earth, plot=False, minimum=None):
    """
    경로 상 bottleneck 링크 찾기
    """
    bottleneck = [[], [], [], []]

    for GT in earth.gateways:
        if GT.name == path[0][0]:
            bottleneck[0].append(str(path[0][0].split(",")[0]) + "," + str(path[1][0]))
            bottleneck[1].append(GT.dataRate)
            bottleneck[2].append(GT.latitude)
            if minimum:
                bottleneck[3].append(minimum / GT.dataRate)

    for i, step in enumerate(path[1:], 1):
        for orbit in earth.LEO:
            for satellite in orbit.sats:
                if satellite.ID == step[0]:
                    for sat in satellite.interSats:
                        if sat[1].ID == path[i + 1][0]:
                            bottleneck[0].append(str(path[i][0]) + "," + str(path[i + 1][0]))
                            bottleneck[1].append(sat[2])
                            bottleneck[2].append(satellite.latitude)
                            if minimum:
                                bottleneck[3].append(minimum / sat[2])

                    for sat in satellite.intraSats:
                        if sat[1].ID == path[i + 1][0]:
                            bottleneck[0].append(str(path[i][0]) + "," + str(path[i + 1][0]))
                            bottleneck[1].append(sat[2])
                            bottleneck[2].append(satellite.latitude)
                            if minimum:
                                bottleneck[3].append(minimum / sat[2])

    for GT in earth.gateways:
        if GT.name == path[-1][0]:
            bottleneck[0].append(str(path[-2][0]) + "," + str(path[-1][0].split(",")[0]))
            bottleneck[1].append(GT.linkedSat[1].downRate)
            bottleneck[2].append(GT.latitude)
            if minimum:
                bottleneck[3].append(minimum / GT.dataRate)

    if plot:
        earth.plotMap(True, True, path, bottleneck)

    minimum = np.amin(bottleneck[1])
    return bottleneck, minimum