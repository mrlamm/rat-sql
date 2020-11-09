import attr
import pyrsistent
import torch

from ratsql.models.head_corner.tree_traversal import TreeTraversal
from ratsql.utils import vocab


class InferenceTreeTraversal(TreeTraversal):

    SIMPLE_TERMINAL_TYPES = {
        'str': str,
        'int': int,
        'float': float,
        'bool': lambda n: {'True': True, 'False': False}.get(n, False),
    }

    SIMPLE_TERMINAL_TYPES_DEFAULT = {
        'str': '',
        'int': 0,
        'float': 0,
        'bool': True,
    }

    def __init__(self, model, desc_enc, example=None):
        super().__init__(model, desc_enc)
        self.example = example

    def clone(self):
        super_clone = super().clone()
        super_clone.example = self.example
        return super_clone

    def rule_choice(self, node_type, goal_type, rule_logits):
        return self.model.rule_infer(node_type, goal_type, rule_logits, self.current_state)

    def token_choice(self, output, gen_logodds):
        return self.model.token_infer(output, gen_logodds, self.desc_enc)

    def pointer_choice(self, node_type, logits, attention_logits):
        # Group them based on pointer map
        pointer_logprobs = self.model.pointer_infer(node_type, logits)
        pointer_map = self.desc_enc.pointer_maps.get(node_type)
        if not pointer_map:
            return pointer_logprobs

        pointer_logprobs = dict(pointer_logprobs)
        return [
            (orig_index, torch.logsumexp(
                torch.stack(
                    tuple(pointer_logprobs[i] for i in mapped_indices),
                    dim=0),
                dim=0))
            for orig_index, mapped_indices in pointer_map.items()
        ]

    def update_using_last_choice(self, last_choice, extra_choice_info, attention_offset):
        super().update_using_last_choice(last_choice, extra_choice_info, attention_offset)

    def finalize(self):
        assert self.current_state == TreeTraversal.State.DONE
        root = self.convert_head_corner_to_node_rep(self.head_corners[-1])

        return root, self.model.preproc.grammar.unparse(root, self.example)
