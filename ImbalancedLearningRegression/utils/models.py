## Standard Library Dependencies
from enum   import Enum, unique
from typing import TypedDict

# Enums Declarations
@unique
class SAMPLE_METHOD(Enum):
    BALANCE = "balance"
    EXTREME = "extreme"

@unique
class RELEVANCE_METHOD(Enum):
    AUTO   = "auto"
    MANUAL = "manual"

@unique
class RELEVANCE_XTRM_TYPE(Enum):
    HIGH = "high"
    BOTH = "both"
    LOW  = "low"

# Typed Dictionaries
class RelevanceParameters(TypedDict):
    method:   RELEVANCE_METHOD
    num_pts:  int
    ctrl_pts: list[float]