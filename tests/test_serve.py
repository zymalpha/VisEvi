"""Unit tests: serving layer — deterministic rendering of a frozen session."""
import copy
import unittest

from visevi.serve import evidence_card, to_prompt_block

SESSION = {
    "image_ref": "D:/x/img.png",
    "demand": {"proposition": "是否存在心脏增大表现之证据", "requirements": [], "priority": 1},
    "graph": {"nodes": [
        {"id": "obs-0001", "category": "observation", "evidence_type": "measurement",
         "finding_or_anatomy": "CTR_mask_caliber",
         "value": {"CTR": 0.3829, "heart_max_width_px": 749, "method": "proxy"},
         "provenance": [], "validity_conditions": []},
        {"id": "claim-0001", "category": "claim", "evidence_type": "claim",
         "finding_or_anatomy": "心胸比增大表现",
         "value": {"proposition": "心胸比增大表现", "bel": 0.8637, "pl": 1.0,
                   "interval_width": 0.1363, "conflict_k": 0.0,
                   "supported_direction": "absent", "combination_rule": "dempster"},
         "provenance": ["obs-0001"], "validity_conditions": ["view assumed PA"]}],
        "edges": []},
    "verdict": {"answerable": True, "answer": "未检出心胸比增大表现",
                "evidence_refs": ["claim-0001"]}}


class TestServe(unittest.TestCase):
    def test_card_contains_claim_chain(self):
        card = evidence_card(SESSION)
        for s in ("claim-0001", "absent", "Bel", "0.3829", "view assumed PA",
                  "obs-0001", "组合≠诊断", "未检出心胸比增大表现"):
            self.assertIn(s, card)
        self.assertNotIn("method=proxy", card)          # verbose fields dropped

    def test_card_deterministic(self):
        self.assertEqual(evidence_card(copy.deepcopy(SESSION)),
                         evidence_card(copy.deepcopy(SESSION)))

    def test_card_empty_session(self):
        s = copy.deepcopy(SESSION)
        s["graph"]["nodes"] = []
        s["verdict"] = {"answerable": False, "abstain_reason": "证据不足", "evidence_refs": []}
        card = evidence_card(s)
        self.assertIn("无可交付证据声明", card)
        self.assertIn("拒答", card)
        self.assertIn("证据不足", card)

    def test_prompt_block_fenced(self):
        self.assertTrue(to_prompt_block(SESSION).startswith("```markdown"))


if __name__ == "__main__":
    unittest.main()
