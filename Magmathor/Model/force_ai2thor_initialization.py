# this will force ai2thor to initialize and download the required assets
import json
import sys
import time
import traceback

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering, Linux64

start_time = time.time()
controller = None
try:
    controller = Controller(
        agentMode="default",
        visibilityDistance=5,
        scene="FloorPlan1",
        # step sizes
        gridSize=0.25,
        snapToGrid=False,
        # image modalities
        renderDepthImage=False,
        renderInstanceSegmentation=True,
        # camera properties
        width=1024,
        height=1024,
        fieldOfView=60,
        platform=Linux64,  # Use CloudRendering for cloud environments
    )

    # Perform a trivial action to ensure simulation steps work.
    event = controller.step(action="MoveAhead")
    action_success = bool(
        getattr(event, "metadata", {}).get("lastActionSuccess", False)
    )
    position = getattr(event, "metadata", {}).get("agent", {}).get("position", {})

    result = {
        "initialized": True,
        "action": "MoveAhead",
        "action_success": action_success,
        "agent_position": position,
        "elapsed_sec": round(time.time() - start_time, 3),
    }
    print(json.dumps(result))
    # explicit simple line for grepping
    if action_success:
        print("--= AI2THOR initialized and dummy action succeeded =--")
    else:
        print(
            "--= !!! AI2THOR initialized but dummy action FAILED !!! =--",
            file=sys.stderr,
        )
        sys.exit(2)
except Exception as e:
    print(
        f"--= !!! AI2THOR initialization or dummy action failed: {e} !!! =--",
        file=sys.stderr,
    )
    traceback.print_exc()
    sys.exit(1)
finally:
    if controller is not None:
        controller.stop()
