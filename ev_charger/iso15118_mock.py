"""Mock ISO 15118 message exchange (Plug & Charge handshake).

Lightweight dataclass request/response messages and a sequential exchange
driver. Each message serializes to bytes (JSON payload) so it can be pushed
through the VLC physical layer in the V2X simulation and reconstructed at the
receiver.
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
from dataclasses import asdict, dataclass, field


@dataclass
class Iso15118Message:
    """Base message: name + payload dict, serializable to/from bytes."""

    name: str
    payload: dict = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        obj = {"name": self.name, "payload": self.payload}
        return json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")

    @classmethod
    def from_bytes(cls, raw: bytes) -> "Iso15118Message":
        obj = json.loads(raw.decode("utf-8"))
        return cls(name=obj["name"], payload=obj["payload"])


def build_session_sequence(evcc_id: str = "EV-ABC123",
                           contract_id: str = "DE-PNC-0001") -> list[Iso15118Message]:
    """The canonical Plug & Charge message sequence (request/response pairs)."""
    return [
        Iso15118Message("SupportedAppProtocolReq",
                        {"protocols": ["urn:iso:15118:2:2013:MsgDef"]}),
        Iso15118Message("SupportedAppProtocolRes", {"responseCode": "OK_SuccessfulNegotiation"}),
        Iso15118Message("SessionSetupReq", {"evccId": evcc_id}),
        Iso15118Message("SessionSetupRes", {"responseCode": "OK", "sessionId": "0A1B2C3D"}),
        Iso15118Message("ServiceDiscoveryReq", {"serviceScope": "EnergyTransfer"}),
        Iso15118Message("ServiceDiscoveryRes",
                        {"responseCode": "OK", "services": ["AC_single_phase", "DC_extended"]}),
        Iso15118Message("PaymentDetailsReq",
                        {"contractId": contract_id, "method": "Contract"}),
        Iso15118Message("PaymentDetailsRes", {"responseCode": "OK", "genChallenge": "Q1W2E3"}),
        Iso15118Message("PowerDeliveryReq", {"chargeProgress": "Start"}),
        Iso15118Message("PowerDeliveryRes", {"responseCode": "OK", "evseStatus": "Ready"}),
        Iso15118Message("ChargingStatusReq", {}),
        Iso15118Message("ChargingStatusRes",
                        {"responseCode": "OK", "evsePresentCurrent": 125, "evsePresentVoltage": 400}),
        Iso15118Message("SessionStopReq", {"chargingSession": "Terminate"}),
        Iso15118Message("SessionStopRes", {"responseCode": "OK"}),
    ]


def serialize_sequence(messages: list[Iso15118Message]) -> bytes:
    """Frame a sequence as length-prefixed messages into one byte stream."""
    out = bytearray()
    for msg in messages:
        body = msg.to_bytes()
        out += len(body).to_bytes(2, "big")  # 2-byte length prefix
        out += body
    return bytes(out)


def deserialize_sequence(stream: bytes) -> list[Iso15118Message]:
    """Inverse of :func:`serialize_sequence`. Tolerant of trailing padding."""
    messages = []
    i = 0
    n = len(stream)
    while i + 2 <= n:
        length = int.from_bytes(stream[i:i + 2], "big")
        i += 2
        if length == 0 or i + length > n:
            break
        body = stream[i:i + length]
        i += length
        try:
            messages.append(Iso15118Message.from_bytes(body))
        except (UnicodeDecodeError, json.JSONDecodeError):
            break
    return messages


def main() -> None:
    msgs = build_session_sequence()
    stream = serialize_sequence(msgs)
    recovered = deserialize_sequence(stream)
    print("ISO 15118 mock (Plug & Charge)")
    print(f"  messages in sequence : {len(msgs)}")
    print(f"  serialized size      : {len(stream)} bytes")
    print(f"  round-trip recovered : {len(recovered)} messages")
    print(f"  lossless round-trip  : {recovered == msgs}")
    print("  sequence:")
    for m in msgs:
        print(f"    - {m.name}")


if __name__ == "__main__":
    main()
