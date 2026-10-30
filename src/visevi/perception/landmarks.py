"""A5: lung apex, base and costophrenic-angle landmarks (CheXmask-U HybridGNet)."""
from __future__ import annotations

import numpy as np

from ..backends import Backend
from .model_tools import ModelTool, Output, image_size, inside_image, int_param, unit

# Node semantics of the LH-120 graph, frozen 2026-10-30 after checking that each
# index is the corresponding geometric extreme on 20 images from ChestX-Det,
# Montgomery, JSRT and NIH (apex/costophrenic 20/20, cardiophrenic 19/20 and 18/20).
# Right lung nodes lie on the image left in standard PA display.
SEMANTIC_NODES = {
    'right_lung_apex': 0, 'right_costophrenic_angle': 21, 'right_cardiophrenic_angle': 29,
    'left_lung_apex': 44, 'left_costophrenic_angle': 65, 'left_cardiophrenic_angle': 71,
}
BASE_CONTOURS = {'right_lung_base_contour': (21, 30), 'left_lung_base_contour': (65, 72)}
NODE_GROUPS = {'right_lung': (0, 44), 'left_lung': (44, 94), 'heart': (94, 120)}
SEMANTIC_TABLE_VERSION = 1


class LocateLungLandmarks(ModelTool):
    name = 'locate_lung_landmarks'
    capability = 'A5'
    description = ('lung apices, costophrenic and cardiophrenic angles and lung-base contours from '
                   'CheXmask-U HybridGNet v1_skip (120-node lung/heart contour); original pixel frame')
    backend = Backend('chexmask-u', 'hybridgnet', {'adapter': 1, 'semantic_table': SEMANTIC_TABLE_VERSION},
                      source='chexmask-u')
    extra_properties = {
        'uncertainty_samples': {'type': 'integer', 'minimum': 0, 'maximum': 100,
                                'description': 'latent samples for per-node spread (0 = off)'},
        'seed': {'type': 'integer', 'minimum': 0},
    }

    def backend_params(self, session, params):
        return {'uncertainty_samples': int_param(params, 'uncertainty_samples', 0, 0, 100),
                'seed': int_param(params, 'seed', 0, 0, 2**31 - 1)}

    def convert(self, session, run, params):
        result = run['result']
        nodes = np.asarray(result['nodes_xy'], dtype=float)
        if nodes.shape != (120, 2) or not np.isfinite(nodes).all():
            raise ValueError('backend returned malformed landmark nodes')
        width, height = image_size(session)
        spread = result.get('sampling', {}).get('node_std_xy')
        common = {'model': 'CheXmask-U HybridGNet v1_skip', 'semantic_table_version': SEMANTIC_TABLE_VERSION,
                  'coordinate_convention': result['coordinate_convention'], 'frame': result['frame']}
        units = []
        for name, index in SEMANTIC_NODES.items():
            xy = nodes[index]
            visible = inside_image(xy, width, height)
            value = {'point_xy': [float(xy[0]), float(xy[1])], 'node_index': index,
                     'status': 'ok' if visible else 'outside_image',
                     'node_std_xy': spread[index] if spread else None, **common}
            units.append(unit(
                name, 'landmark', value,
                region={'x0': float(xy[0]), 'y0': float(xy[1]), 'x1': float(xy[0]), 'y1': float(xy[1])},
                provenance=[f'cached:CheXmask-U:{run["binding"]["image_sha256"][:16]}'],
                validity_conditions=['patient side follows the model node groups (right lung = image left)',
                                     'point lies outside the image; not localized'] if not visible else
                                    ['patient side follows the model node groups (right lung = image left)']))
        for name, (start, stop) in BASE_CONTOURS.items():
            units.append(unit(
                name, 'landmark',
                {'polyline_xy': nodes[start:stop].tolist(), 'node_range': [start, stop],
                       'description': 'lung base from costophrenic to cardiophrenic angle (diaphragm interface)',
                       **common},
                provenance=[f'cached:CheXmask-U:{run["binding"]["image_sha256"][:16]}'],
                validity_conditions=['contour of the predicted lung outline; hemidiaphragm dome not separately localized']))
        units.append(unit(
            'lung_heart_contour_nodes', 'landmark',
            {'nodes_xy': nodes.tolist(), 'node_groups': {k: list(v) for k, v in NODE_GROUPS.items()},
                   'node_std_xy': spread, 'sampling': result.get('sampling'), **common},
            provenance=[f'cached:CheXmask-U:{run["binding"]["image_sha256"][:16]}'],
            validity_conditions=['full predicted contour; only the semantic nodes above have frozen meanings']))
        return Output(units=units, landmarks={'lm_px': nodes, 'groups': dict(NODE_GROUPS),
                                              'semantic_nodes': dict(SEMANTIC_NODES), 'source': self.name})
