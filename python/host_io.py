
import numpy as np

import ifa7_config as C


def pack_inputs(Q, K, V):
    """Q,K,V : int arrays (N,DK) in INT8 range -> bytes for the FPGA."""
    def to_bytes(M):
        a = np.asarray(M).astype(np.int64).flatten()
        assert a.size == C.N * C.DK
        return bytes(((int(x) & 0xFF) for x in a))   # two's-complement byte
    return to_bytes(Q) + to_bytes(K) + to_bytes(V)


def unpack_output(buf):
    """buf : bytes from the FPGA -> O int array (N,DK), Q(.,OUT_F)."""
    nbytes = C.N * C.DK * 3
    assert len(buf) >= nbytes, f"need {nbytes} bytes, got {len(buf)}"
    O = np.zeros(C.N * C.DK, dtype=np.int64)
    for i in range(C.N * C.DK):
        b0, b1, b2 = buf[3 * i], buf[3 * i + 1], buf[3 * i + 2]   # big-endian
        val = (b0 << 16) | (b1 << 8) | b2
        if val & (1 << (C.OUT_W - 1)):       # sign-extend 24-bit
            val -= (1 << C.OUT_W)
        O[i] = val
    return O.reshape(C.N, C.DK)


def run_on_board(Q, K, V, port, baud=C.BAUD, timeout=30.0):
    """Send Q,K,V over `port` (e.g. 'COM5') and return the FPGA's O matrix.
    Requires `pyserial` (pip install pyserial)."""
    import serial   # local import so the module works without pyserial
    tx = pack_inputs(Q, K, V)
    nrx = C.N * C.DK * 3
    with serial.Serial(port, baud, timeout=timeout) as s:
        s.reset_input_buffer()
        s.write(tx)
        s.flush()
        buf = bytearray()
        while len(buf) < nrx:
            chunk = s.read(nrx - len(buf))
            if not chunk:
                raise TimeoutError(f"received {len(buf)}/{nrx} bytes before timeout")
            buf.extend(chunk)
    return unpack_output(bytes(buf))


if __name__ == "__main__":
    # round-trip pack/unpack self-check against the golden model
    import ifa7_golden as G
    rs = np.random.RandomState(0xA7)
    Q = rs.randint(-6, 7, (C.N, C.DK)).astype(np.int64)
    K = rs.randint(-6, 7, (C.N, C.DK)).astype(np.int64)
    V = rs.randint(-128, 128, (C.N, C.DK)).astype(np.int64)
    O = G.attention_fixed(Q, K, V)
    # simulate the wire: pack O the way the FPGA would, then unpack
    wire = bytearray()
    for x in O.flatten():
        v = int(x) & ((1 << C.OUT_W) - 1)
        wire += bytes([(v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF])
    O2 = unpack_output(bytes(wire))
    print("host_io pack/unpack round-trip OK:", np.array_equal(O, O2))
    print("input stream bytes:", len(pack_inputs(Q, K, V)),
          " output stream bytes:", len(wire))
