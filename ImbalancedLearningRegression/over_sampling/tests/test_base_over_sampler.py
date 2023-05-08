## Third Party Dependencies
from pytest import raises
from pandas import DataFrame

## Standard Library Dependencies
from typing        import Any
from unittest.mock import patch, _patch

## Internal Dependencies
from ImbalancedLearningRegression.over_sampling.base  import BaseOverSampler

def test_fit_resample() -> None:
    # Create BaseOverSampler
    base_oversampler, p = create_baseoversampler()

    with raises(NotImplementedError):
        base_oversampler.fit_resample(data = DataFrame(), response_variable = "foobar")

    destroy_baseoversampler(p)

def test_oversampler() -> None:
    # Create BaseOverSampler
    base_oversampler, p = create_baseoversampler()

    with raises(NotImplementedError):
        base_oversampler._oversample(data = DataFrame(), indices = {}, perc = [])

    destroy_baseoversampler(p)

# Helper Functions
def create_baseoversampler() -> tuple[BaseOverSampler, "_patch[Any]"]:
    p = patch.multiple(BaseOverSampler, __abstractmethods__=set())
    p.start()
    base_sampler = BaseOverSampler() # type: ignore
    return base_sampler, p 

def destroy_baseoversampler(p: "_patch[Any]") -> None:
    p.stop()