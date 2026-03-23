import matplotlib.pyplot as plt
import math

def live_plot_simulation(env, earth, simulationTimelimit, dt=10, path_pair=(0, 1)):
    # 실험 루프용으로는 dt=0.5 설정 필요 (dt=10은 데모용)
    """
    env를 조금씩 진행시키면서 지도를 실시간 갱신한다.
    path_pair: (source_gateway_index, destination_gateway_index)
    dt: 한 번에 진행할 simulation time
    """
    plt.ion()
    fig = plt.figure(figsize=(14, 7))

    prev_positions = None

    while env.now < simulationTimelimit:
        next_time = min(env.now + dt, simulationTimelimit)
        env.run(until=next_time)
        sat = earth.LEO[0].sats[0]
        print(
            f"time={env.now:.1f}, sat={sat.ID}, "
            f"lat={math.degrees(sat.latitude):.4f}, "
            f"lon={math.degrees(sat.longitude):.4f}"
        )   

        plt.clf()

        path = None
        src_idx, dst_idx = path_pair
        if src_idx is not None and dst_idx is not None and len(earth.gateways) > max(src_idx, dst_idx):
            src = earth.gateways[src_idx]
            dst = earth.gateways[dst_idx]
            path = src.paths.get(dst.name, None)

        earth.plotMap(True, True, path=path)

        current_positions = []
        for plane in earth.LEO:
            for sat in plane.sats:
                x = int((0.5 + math.degrees(sat.longitude) / 360) * 1440)
                y = int((0.5 - math.degrees(sat.latitude) / 180) * 720)
                current_positions.append((x, y))

        if prev_positions is not None:
            xs = [p[0] for p in prev_positions]
            ys = [p[1] for p in prev_positions]
            plt.scatter(xs, ys, s=10, alpha=0.2, c='white')

        prev_positions = current_positions

        plt.title(f"Simulation time = {env.now:.1f}")
        plt.pause(0.2)

    plt.ioff()
    plt.show()