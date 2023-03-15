from enum import Enum, unique

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