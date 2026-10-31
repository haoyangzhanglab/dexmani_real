"""Grounded-SAM2 process entry; stdout is reserved for framed binary replies."""

import argparse
import os
import sys
import traceback

from .segmentation_protocol import receive, send


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sam-config", required=True)
    parser.add_argument("--dino-weights", required=True)
    args = parser.parse_args()
    # Native libraries also write fd 1; redirect it before importing any models.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    engine = None
    session = None
    last_request = 0
    try:
        while True:
            try:
                request, arrays = receive(sys.stdin.buffer)
            except EOFError:
                break
            reply = {key: request[key] for key in ("session", "request", "frame_key")}
            try:
                if session is None:
                    session = request["session"]
                if request["session"] != session or request["request"] != last_request + 1:
                    raise ValueError("Out-of-order SAM2 request")
                last_request = request["request"]
                operation = request["operation"]
                output = {}
                if operation == "load":
                    if engine is not None:
                        raise ValueError("Model already loaded")
                    from .sam2_engine import SAM2Engine

                    engine = SAM2Engine(
                        args.sam_root, args.checkpoint, args.sam_config, args.dino_weights
                    )
                    reply["provenance"] = engine.provenance
                elif engine is None:
                    raise ValueError("Load model first")
                elif operation == "predict":
                    output, reply["labels"] = engine.predict(
                        arrays["rgb"],
                        request["text"],
                        request["box_threshold"],
                        request["text_threshold"],
                    )
                    reply["provenance"] = engine.provenance
                elif operation == "initialize":
                    output["masks"] = engine.initialize(
                        arrays["rgb"], request["ids"], arrays["masks"]
                    )
                    reply["ids"] = engine.ids
                elif operation == "track":
                    output["masks"] = engine.track(arrays["rgb"])
                    reply["ids"] = engine.ids
                else:
                    raise ValueError(f"Unknown operation: {operation}")
                send(protocol, reply, output)
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                send(protocol, dict(**reply, error=f"{type(error).__name__}: {error}"))
                return 1
    finally:
        protocol.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
