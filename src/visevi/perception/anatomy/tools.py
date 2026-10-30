"""A1-A4 interfaces with explicit live-inference and cache-consumption modes."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ...core.spec import ToolSpec, parameter_schema, evidence_result_schema
from ...core.units import ToolRefusal
from . import ANATOMY_GROUPS, SegmentAnatomy
from .pspnet import PSPNetPredictor, file_sha256


@lru_cache(maxsize=2)
def _predictor(weights_dir: str, device: str):
    return PSPNetPredictor(weights_dir, device=device)


class AnatomyTool:
    def __init__(self, name: str):
        self.name = name
        self.capability, self.organs = ANATOMY_GROUPS[name]

    def describe(self) -> ToolSpec:
        schema = parameter_schema({
            'masks_dir': {'type': 'string'}, 'output_dir': {'type': 'string'},
            'image_ref': {'type': 'string'}, 'weights_dir': {'type': 'string'},
            'device': {'type': 'string'},
        })
        schema['oneOf'] = [
            {'required': ['masks_dir'], 'not': {'required': ['output_dir']}},
            {'required': ['output_dir'], 'not': {'required': ['masks_dir']}},
        ]
        return ToolSpec(name=self.name, category='perception',
                        description=f'{self.capability}: PSPNet {self.organs}; supply masks_dir for a verified cache (image, model, weights, preprocessing) OR output_dir for live inference',
                        input_schema=schema, output_schema=evidence_result_schema())

    def run(self, session, **params):
        unknown = set(params) - self.describe().input_schema['properties'].keys()
        if unknown or any(not isinstance(value, str) or not value for value in params.values()):
            return ToolRefusal(tool=self.name, reason=f'invalid parameters: {sorted(unknown)}; all supplied values must be nonempty strings')
        if bool(params.get('masks_dir')) == bool(params.get('output_dir')):
            return ToolRefusal(tool=self.name, reason='provide exactly one of masks_dir (cached) or output_dir (live)')
        if 'masks_dir' in params and {'weights_dir', 'device'} & params.keys():
            return ToolRefusal(tool=self.name, reason='weights_dir/device apply only to live inference (output_dir)')
        image_ref = params.get('image_ref', session.image_ref)
        try:
            if Path(image_ref).resolve() != Path(session.image_ref).resolve():
                raise ValueError('image_ref must match the session image')
            mode = 'cached_artifact'
            cache_hit = None
            if 'masks_dir' in params:
                masks_dir = params['masks_dir']
            else:
                mode = 'live_inference'
                masks_dir = params['output_dir']
                cache_hit = (Path(masks_dir) / 'cache_record.json').exists()
                default_weights = Path(__file__).resolve().parents[4] / 'tools' / 'weights'
                predictor = _predictor(str(Path(params.get('weights_dir', default_weights)).resolve()),
                                       params.get('device', 'cpu'))
                predictor.predict(image_ref, 'a-tools:' + file_sha256(image_ref), masks_dir)
            output = SegmentAnatomy().run(session, masks_dir=masks_dir,
                                         image_ref=image_ref, organs=self.organs)
            if isinstance(output, ToolRefusal):
                output.tool = self.name
                return output
            for unit in output:
                unit.source_tool = self.name
                unit.value.update(capability=self.capability, execution_mode=mode,
                                  prediction_cache_hit=cache_hit)
                unit.validity_conditions.append('model prediction; no independent accuracy acceptance yet')
            return output
        except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as exc:
            return ToolRefusal(tool=self.name, reason=str(exc))


def build_anatomy_tools():
    return [AnatomyTool(name) for name in ANATOMY_GROUPS]
