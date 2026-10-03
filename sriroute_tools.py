"""
sriroute_tools.py
====================
The SriRoute Dispatch tools (PROVIDED -- do not modify).

Four tools your ReAct agent can call. Each tool has a matching Pydantic
argument model -- your harness validates the LLM's proposed tool call
against these BEFORE executing it (see react_loop_lab.py).

Note: assign_courier does not itself check that the vehicle has enough
capacity or can arrive within the promised window -- it will happily
"succeed" at assigning an unsuitable vehicle if asked to. Tools are dumb
executors; verifying that a call is *sensible* before making it is the
agent's job, not the tool's. Keep this in mind for the exit question.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------
# Mock backend data
# ---------------------------------------------------------------------

DELIVERY_REQUESTS = {
    "REQ-901": {
        "customer": "Meera", "pickup": "T Nagar", "dropoff": "Adyar",
        "package_type": "standard", "weight_kg": 4,
        "zone": "South Chennai", "promised_window": "2:00-3:00 PM",
    },
    "REQ-902": {
        "customer": "Farhan", "pickup": "OMR Warehouse", "dropoff": "Sholinganallur",
        "package_type": "perishable", "weight_kg": 12,
        "zone": "OMR", "promised_window": "1:00-1:30 PM",
    },
    "REQ-903": {
        "customer": "Divya", "pickup": "Sri City Hub", "dropoff": "Tada",
        "package_type": "standard", "weight_kg": 25,
        "zone": "Sri City", "promised_window": "4:00-6:00 PM",
    },
    "REQ-904": {
        "customer": "Karthik", "pickup": "Sri City Hub", "dropoff": "Tada",
        "package_type": "standard", "weight_kg": 5,
        "zone": "Sri City", "promised_window": "11:00-11:15 AM",
    },
}

FLEET = {
    "South Chennai": [
        {"vehicle_id": "BIKE-12", "vehicle_type": "bike", "capacity_kg": 8, "eta_minutes": 10, "priority": False, "surge_cost": 0.0},
        {"vehicle_id": "VAN-04", "vehicle_type": "van", "capacity_kg": 50, "eta_minutes": 18, "priority": False, "surge_cost": 0.0},
    ],
    "OMR": [
        # No refrigerated vehicle available in this zone -- this is what
        # makes REQ-902 (a perishable item) infeasible.
        {"vehicle_id": "BIKE-07", "vehicle_type": "bike", "capacity_kg": 8, "eta_minutes": 6, "priority": False, "surge_cost": 0.0},
    ],
    "Sri City": [
        {"vehicle_id": "VAN-09", "vehicle_type": "van", "capacity_kg": 60, "eta_minutes": 25, "priority": False, "surge_cost": 0.0},
        {"vehicle_id": "PRIORITY-VAN-01", "vehicle_type": "van", "capacity_kg": 60, "eta_minutes": 8, "priority": True, "surge_cost": 450.0},
    ],
}


# ---------------------------------------------------------------------
# Pydantic argument models -- one per tool, plus one for the "finish"
# action. Your harness validates the LLM's proposed action_input against
# these before calling anything.
# ---------------------------------------------------------------------

class GetDeliveryRequestArgs(BaseModel):
    request_id: str


class CheckFleetAvailabilityArgs(BaseModel):
    zone: str
    vehicle_type: Optional[Literal["bike", "van"]] = None


class AssignCourierArgs(BaseModel):
    request_id: str
    vehicle_id: str


class RequestSurgeApprovalArgs(BaseModel):
    request_id: str
    extra_cost: float = Field(gt=0)
    reason: str


class FinishArgs(BaseModel):
    final_answer: str


TOOL_ARG_MODELS = {
    "get_delivery_request": GetDeliveryRequestArgs,
    "check_fleet_availability": CheckFleetAvailabilityArgs,
    "assign_courier": AssignCourierArgs,
    "request_surge_approval": RequestSurgeApprovalArgs,
    "finish": FinishArgs,
}


# ---------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------

def get_delivery_request(request_id: str) -> dict:
    return DELIVERY_REQUESTS.get(request_id, {"error": f"{request_id} not found"})


def check_fleet_availability(zone: str, vehicle_type: Optional[str] = None) -> dict:
    vehicles = FLEET.get(zone, [])
    if vehicle_type is not None:
        vehicles = [v for v in vehicles if v["vehicle_type"] == vehicle_type]
    if not vehicles:
        return {"zone": zone, "vehicles": [], "note": "No matching vehicles available in this zone."}
    return {"zone": zone, "vehicles": vehicles}


def assign_courier(request_id: str, vehicle_id: str) -> dict:
    # Deliberately does not validate capacity, ETA, or vehicle existence
    # -- see the module docstring.
    return {"status": "SUCCESS", "message": f"Assigned {vehicle_id} to {request_id}."}


def default_console_approval(request_id: str, extra_cost: float, reason: str) -> bool:
    answer = input(
        f"[GUARDRAIL] Agent wants to spend an extra Rs.{extra_cost:.2f} on {request_id} "
        f"({reason}). Approve? (y/n): "
    )
    return answer.strip().lower().startswith("y")


def make_request_surge_approval(approval_fn=default_console_approval):
    """
    Returns a request_surge_approval tool function bound to the given
    approval_fn. In tests, pass a fixed True/False function instead of
    default_console_approval so you don't have to sit at a keyboard.
    """
    def request_surge_approval(request_id: str, extra_cost: float, reason: str) -> dict:
        approved = approval_fn(request_id, extra_cost, reason)
        if approved:
            return {"status": "APPROVED", "request_id": request_id, "extra_cost": extra_cost}
        return {
            "status": "DENIED", "request_id": request_id,
            "note": "Supervisor denied the surge request. Use the standard fleet "
                    "and notify the customer of a possible delay.",
        }
    return request_surge_approval


def make_tool_registry(approval_fn=None):
    """The full set of callable tools, keyed by name (matches TOOL_ARG_MODELS
    minus 'finish', which your harness handles specially)."""
    surge_fn = make_request_surge_approval(approval_fn) if approval_fn else make_request_surge_approval()
    return {
        "get_delivery_request": get_delivery_request,
        "check_fleet_availability": check_fleet_availability,
        "assign_courier": assign_courier,
        "request_surge_approval": surge_fn,
    }
