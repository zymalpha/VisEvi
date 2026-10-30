"""Shared plumbing for model-backed A tools (A5–A8, A12 auto, A18, A24–A26).

Each tool runs its backend through ``visevi.backends.run_cached`` (isolated
worker, verified weights, bound cache), then converts the worker result into
EvidenceUnits in the original image pixel frame. Model outputs are predictions:
confidence stays ``None`` unless the backend gives a calibrated-free raw score,
which is reported as ``raw_score`` instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.spec import ToolSpec, parameter_schema, evidence_result_schema
from ..core.units import EvidenceUnit, ToolRefusal
from .. import backends
from ..backends import Backend


COMMON_PROPERTIES = {
    'output_dir': {'type': 'string', 'description': 'new directory, or one holding an identical bound cache'},
    'device': {'type': 'string', 'description': "e.g. 'cuda:4'; default from $VISEVI_MODEL_DEVICE or 'cpu'"},
}


@dataclass
class Output:
    """Units are committed atomically; masks/landmarks attach only after ids exist."""
    units: list = field(default_factory=list)
    masks: list = field(default_factory=list)       # (name, bool array, metadata, unit)
    landmarks: dict | None = None


def unit(finding: str, evidence_type: str, value: dict, **kwargs) -> EvidenceUnit:
    """Uncommitted observation; ``ModelTool.run`` assigns the id."""
    return EvidenceUnit(id='', category='observation', evidence_type=evidence_type,
                        finding_or_anatomy=finding, value=value, **kwargs)


class ModelTool:
    """Base class; subclasses set name/capability/backend and implement ``convert``."""

    name: str
    capability: str
    description: str
    backend: Backend
    extra_properties: dict = {}
    extra_required: tuple[str, ...] = ()

    def describe(self) -> ToolSpec:
        schema = parameter_schema({**COMMON_PROPERTIES, **self.extra_properties},
                                  ('output_dir', *self.extra_required))
        return ToolSpec(name=self.name, category='perception',
                        description=f'{self.capability}: {self.description}',
                        input_schema=schema, output_schema=evidence_result_schema())

    # -- hooks --------------------------------------------------------------
    def backend_params(self, session, params: dict) -> dict:
        """Validated parameters forwarded to the worker (part of the cache binding)."""
        return {}

    def convert(self, session, run: dict, params: dict) -> Output | ToolRefusal:
        raise NotImplementedError

    # -- protocol -------------------------------------------------------------
    def refuse(self, reason: str, **kwargs) -> ToolRefusal:
        return ToolRefusal(tool=self.name, reason=reason, **kwargs)

    def run(self, session, **params):
        properties = self.describe().input_schema['properties']
        unknown = set(params) - properties.keys()
        missing = {'output_dir', *self.extra_required} - params.keys()
        if unknown or missing:
            return self.refuse(f'invalid parameters: unexpected={sorted(unknown)}, missing={sorted(missing)}')
        for key in ('output_dir', 'device'):
            if key in params and (not isinstance(params[key], str) or not params[key]):
                return self.refuse(f'{key} must be a nonempty string')
        device = params.get('device') or backends.default_device()
        try:
            worker_params = self.backend_params(session, params)
            if isinstance(worker_params, ToolRefusal):
                return worker_params
            run = backends.run_cached(self.backend, session.image_ref, params['output_dir'],
                                      device, worker_params)
            output = self.convert(session, run, params)
        except (OSError, ValueError, TypeError, KeyError, backends.WorkerError) as exc:
            return self.refuse(f'{type(exc).__name__}: {exc}')
        if isinstance(output, ToolRefusal):
            return output
        execution = {'backend': self.backend.name, 'cache_hit': run['cache_hit'],
                     'output_dir': str(Path(params['output_dir']).resolve()), 'device': device,
                     'weights_revision': run['binding']['weights']['revision']}
        for item in output.units:
            item.id = session.next_id('obs')
            item.source_tool = self.name
            item.value.setdefault('capability', self.capability)
            item.value['execution'] = dict(execution)
            item.validity_conditions.append('model prediction; no independent accuracy acceptance yet')
            session.add_observation(item)
        for name, mask, metadata, owner in output.masks:
            session.attach_masks({name: mask}, {name: {**metadata, 'observation_id': owner.id}})
        if output.landmarks is not None:
            session.attach_landmarks(output.landmarks)
        return output.units


def image_size(session) -> tuple[int, int]:
    frame = backends.inspect_image(session.image_ref)
    return frame['width'], frame['height']


def inside_image(xy, width: int, height: int) -> bool:
    x, y = xy
    return -0.5 <= x <= width - 0.5 and -0.5 <= y <= height - 0.5


def int_param(params: dict, key: str, default: int, low: int, high: int) -> int:
    value = params.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f'{key} must be an integer in [{low}, {high}]')
    return int(value)
