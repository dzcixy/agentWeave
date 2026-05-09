from agentweaver.profiling.latency_model import LatencyModel
from agentweaver.simulator.nisp import NISP, ParkingState


def test_nisp_hot_warm_cold() -> None:
    n = NISP(LatencyModel())
    hot = n.park_state("pytest", [1, 2, 3], 100, 1000, 10, 10, 10_000_000)
    assert hot.state == ParkingState.HOT
    cold = n.park_state("pytest", [120, 200], 1000, 100, 10, 99, 100)
    assert cold.state == ParkingState.COLD
    warm = n.park_state("pytest", [120, 200], 100, 1000, 10, 80, 100)
    assert warm.state in {ParkingState.WARM, ParkingState.HOT}
