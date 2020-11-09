import collections
import collections.abc
import copy
import csv
import itertools
import json
import os

import attr
import entmax
import enum
import torch
import torch.nn.functional as F

from ratsql.models import abstract_preproc
from ratsql.models import attention
from ratsql.models import variational_lstm
from ratsql.models.nl2code.decoder import NL2CodeDecoderPreproc
from ratsql.models.head_corner.infer_tree_traversal import InferenceTreeTraversal
from ratsql.models.head_corner.train_tree_traversal import TrainTreeTraversal
from ratsql.models.head_corner.tree_traversal import TreeTraversal
from ratsql.utils import registry
from ratsql.utils import serialization
from ratsql.utils import vocab


def lstm_init(device, num_layers, hidden_size, *batch_sizes):
    init_size = batch_sizes + (hidden_size,)
    if num_layers is not None:
        init_size = (num_layers,) + init_size
    init = torch.zeros(*init_size, device=device)
    return (init, init)


def maybe_stack(items, dim=None):
    to_stack = [item for item in items if item is not None]
    if not to_stack:
        return None
    elif len(to_stack) == 1:
        return to_stack[0].unsqueeze(dim)
    else:
        return torch.stack(to_stack, dim)


def accumulate_logprobs(d, keys_and_logprobs):
    for key, logprob in keys_and_logprobs:
        existing = d.get(key)
        if existing is None:
            d[key] = logprob
        else:
            d[key] = torch.logsumexp(
                torch.stack((logprob, existing), dim=0),
                dim=0)


def get_field_presence_info(ast_wrapper, node, field_infos):
    present = []
    for field_info in field_infos:
        field_value = node.get(field_info.name)
        is_present = field_value is not None and field_value != []

        maybe_missing = field_info.opt or field_info.seq
        is_builtin_type = field_info.type in ast_wrapper.primitive_types

        if maybe_missing and is_builtin_type:
            # TODO: make it possible to deal with "singleton?"
            present.append(is_present and type(field_value).__name__)
        elif maybe_missing and not is_builtin_type:
            present.append(is_present)
        #elif not maybe_missing and field_info.type in ["table", "column"]:   ## added this condition...
        #    assert is_present
        #    present.append(True)
        elif not maybe_missing and is_builtin_type:
            present.append(type(field_value).__name__)
        elif not maybe_missing and not is_builtin_type:
            assert is_present
            present.append(True)
    return tuple(present)


def rule_match(string_lhs, rule_string, rule):
    """ Check if string '->' rule form matches with a rule """
    if isinstance(rule[1], str):
        return rule_string == rule[0]+" -> "+rule[1]+"/NULL"
    elif isinstance(rule[1], tuple):
        return string_lhs == rule[0]
    else:
        return string_lhs == rule[0]


def get_rule_string_from_node(node_type, child_node, ast_wrapper):
    if isinstance(child_node, list):
        return node_type, len(child_node)
    elif node_type in ast_wrapper.product_types:
        return None
    elif not node_type:
        return None
    else:
        return node_type, child_node["_type"]


@attr.s
class PredictPreterminal:
    ttype = attr.ib()
    goal_type = attr.ib()

    def __dict__(self):
        return {"ttype": self.ttype,
                "goal_type": self.goal_type,
                "class": "PredictPreterminal"}

@attr.s
class ExpandUp:
    rule = attr.ib()
    goal_type = attr.ib()

    def __dict__(self):
        return {"rule": self.rule,
                "goal_type": self.goal_type,
                "class": "ExpandUp"}

@attr.s
class Point:
    ttype = attr.ib()
    value = attr.ib()

    def __dict__(self):
        return {"value": self.value,
                "ttype": self.ttype,
                "class": "Point"}



def get_rule_match_indices(string_lhs, rule_string, rule_index):
    return [rule_index[rule] for rule in rule_index if rule_match(string_lhs, rule_string, rule)]

@registry.register('decoder', 'HeadCorner')
class HeadCornerDecoder(torch.nn.Module):
    Preproc = NL2CodeDecoderPreproc

    class Handler:
        handlers = {}

        @classmethod
        def register_handler(cls, func_type):
            if func_type in cls.handlers:
                raise RuntimeError(f"{func_type} handler is already registered")

            def inner_func(func):
                cls.handlers[func_type] = func.__name__
                return func

            return inner_func

    class State(enum.Enum):
        EXPAND_UP = 0
        PREDICT_HEAD_CORNER = 1
        POINT = 2

    def __init__(
            self,
            device,
            preproc,
            grammar_path,
            rule_emb_size=128,
            node_embed_size=64,
            # TODO: This should be automatically inferred from encoder
            enc_recurrent_size=256,
            recurrent_size=256,
            dropout=0.,
            desc_attn='bahdanau',
            copy_pointer=None,
            multi_loss_type='logsumexp',
            sup_att=None,
            use_align_mat=False,
            use_align_loss=False,
            enumerate_order=False,
            loss_type="softmax"):
        super().__init__()
        self._device = device
        self.preproc = preproc
        self.ast_wrapper = preproc.ast_wrapper
        self.terminal_vocab = preproc.vocab
        self.preproc.primitive_types.append("singleton")

        if self.preproc.use_seq_elem_rules:
            self.node_type_vocab = vocab.Vocab(
                sorted(self.preproc.primitive_types) +
                sorted(self.ast_wrapper.custom_primitive_types) +
                sorted(self.preproc.sum_type_constructors.keys()) +
                sorted(self.preproc.field_presence_infos.keys()) +
                sorted(self.preproc.seq_lengths.keys()),
                special_elems=())
        else:
            self.node_type_vocab = vocab.Vocab(
                sorted(self.preproc.primitive_types) +
                sorted(self.ast_wrapper.custom_primitive_types) +
                sorted(self.ast_wrapper.sum_types.keys()) +
                sorted(self.ast_wrapper.singular_types.keys()) +
                sorted(self.preproc.seq_lengths.keys()),
                special_elems=())

        self.all_rules, self.rules_index, self.parent_to_preterminal, self.preterminal_mask, self.preterminal_debug, \
        self.preterminal_types, self.parent_to_hc, self.hc_table, self.hc_debug, self.parent_to_head, \
                    self.parent_to_rule = self.compute_rule_masks(grammar_path)


        # json.dump(dict(self.parent_to_preterminal), open('data/spider/head-corner-glove,cv_link=true/p.json'))
        #json.dump({"parent_to_preterminal": dict(self.parent_to_preterminal),
        #           "preterminal_mask": dict(self.preterminal_mask),
        #           "parent_to_hc": {key: sorted(list(self.parent_to_hc[key])) for key in self.parent_to_hc},
        #           "hc_table": {key: dict(self.hc_table[key]) for key in self.hc_table},
        #           "parent_to_head": dict(self.parent_to_head),
        #           "node_type_vocab_e2i": dict(self.node_type_vocab.elem_to_id),
        #           "node_type_vocab_i2e": dict(self.node_type_vocab.id_to_elem),
        #           # "terminal_vocab": self.terminal_vocab,
        #           # "rules_index": self.rules_index,
        #           "parent_to_rule": dict(self.parent_to_rule),
        #            },
        #          open('data/spider/head-corner-glove,cv_link=true/head_corner_elems.json', 'w'))

        self.rule_emb_size = rule_emb_size
        self.node_emb_size = node_embed_size
        self.enc_recurrent_size = enc_recurrent_size
        self.recurrent_size = recurrent_size

        self.use_align_mat = use_align_mat
        self.use_align_loss = use_align_loss
        self.enumerate_order = enumerate_order

        if use_align_mat:
            from ratsql.models.spider import spider_dec_func
            self.compute_align_loss = lambda *args: \
                spider_dec_func.compute_align_loss(self, *args)
            self.compute_pointer_with_align = lambda *args: \
                spider_dec_func.compute_pointer_with_align_head_corner(self, *args)

        self.state_update = variational_lstm.RecurrentDropoutLSTMCell(
            input_size=self.rule_emb_size * 2 + self.enc_recurrent_size + self.recurrent_size * 2 + self.node_emb_size,
            hidden_size=self.recurrent_size,
            dropout=dropout)

        self.attn_type = desc_attn
        if desc_attn == 'bahdanau':
            self.desc_attn = attention.BahdanauAttention(
                query_size=self.recurrent_size,
                value_size=self.enc_recurrent_size,
                proj_size=50)
        elif desc_attn == 'mha':
            self.desc_attn = attention.MultiHeadedAttention(
                h=8,
                query_size=self.recurrent_size,
                value_size=self.enc_recurrent_size)
        elif desc_attn == 'mha-1h':
            self.desc_attn = attention.MultiHeadedAttention(
                h=1,
                query_size=self.recurrent_size,
                value_size=self.enc_recurrent_size)
        elif desc_attn == 'sep':
            self.question_attn = attention.MultiHeadedAttention(
                h=1,
                query_size=self.recurrent_size,
                value_size=self.enc_recurrent_size)
            self.schema_attn = attention.MultiHeadedAttention(
                h=1,
                query_size=self.recurrent_size,
                value_size=self.enc_recurrent_size)
        else:
            # TODO: Figure out how to get right sizes (query, value) to module
            self.desc_attn = desc_attn
        self.sup_att = sup_att
        self.rule_logits = torch.nn.Sequential(
            torch.nn.Linear(self.recurrent_size, self.rule_emb_size),
            torch.nn.Tanh(),
            torch.nn.Linear(self.rule_emb_size, len(self.rules_index)))
        self.rule_embedding = torch.nn.Embedding(
            num_embeddings=len(self.rules_index),
            embedding_dim=self.rule_emb_size)

        self.gen_logodds = torch.nn.Linear(self.recurrent_size, 1)
        self.terminal_logits = torch.nn.Sequential(
            torch.nn.Linear(self.recurrent_size, self.rule_emb_size),
            torch.nn.Tanh(),
            torch.nn.Linear(self.rule_emb_size, len(self.terminal_vocab)))
        self.terminal_embedding = torch.nn.Embedding(
            num_embeddings=len(self.terminal_vocab),
            embedding_dim=self.rule_emb_size)
        if copy_pointer is None:
            self.copy_pointer = attention.BahdanauPointer(
                query_size=self.recurrent_size,
                key_size=self.enc_recurrent_size,
                proj_size=50)
        else:
            # TODO: Figure out how to get right sizes (query, key) to module
            self.copy_pointer = copy_pointer
        if multi_loss_type == 'logsumexp':
            self.multi_loss_reduction = lambda logprobs: -torch.logsumexp(logprobs, dim=1)
        elif multi_loss_type == 'mean':
            self.multi_loss_reduction = lambda logprobs: -torch.mean(logprobs, dim=1)

        self.pointers = torch.nn.ModuleDict()
        self.pointer_action_emb_proj = torch.nn.ModuleDict()
        for pointer_type in self.preproc.grammar.pointers:
            self.pointers[pointer_type] = attention.ScaledDotProductPointer(
                query_size=self.recurrent_size,
                key_size=self.enc_recurrent_size)
            self.pointer_action_emb_proj[pointer_type] = torch.nn.Linear(
                self.enc_recurrent_size, self.rule_emb_size)

        self.node_type_embedding = torch.nn.Embedding(
            num_embeddings=len(self.node_type_vocab),
            embedding_dim=self.node_emb_size)

        # TODO batching
        self.zero_rule_emb = torch.zeros(1, self.rule_emb_size, device=self._device)
        self.zero_recurrent_emb = torch.zeros(1, self.recurrent_size, device=self._device)
        if loss_type == "softmax":
            self.xent_loss = torch.nn.CrossEntropyLoss(reduction='none')
        elif loss_type == "entmax":
            self.xent_loss = entmax.entmax15_loss
        elif loss_type == "sparsemax":
            self.xent_loss = entmax.sparsemax_loss
        elif loss_type == "label_smooth":
            self.xent_loss = self.label_smooth_loss

        self.goals = None
        self.head_corners = None
        self.operation = None

    def label_smooth_loss(self, X, target, smooth_value=0.1):
        if self.training:
            logits = torch.log_softmax(X, dim=1)
            size = X.size()[1]
            one_hot = torch.full(X.size(), smooth_value / (size - 1)).to(X.device)
            one_hot.scatter_(1, target.unsqueeze(0), 1 - smooth_value)
            loss = F.kl_div(logits, one_hot, reduction="batchmean")
            return loss.unsqueeze(0)
        else:
            return torch.nn.functional.cross_entropy(X, target, reduction="none")

    @classmethod
    def _calculate_rules(cls, preproc):
        offset = 0

        all_rules = []
        rules_mask = {}

        # Rules of the form:
        # expr -> Attribute | Await | BinOp | BoolOp | ...
        # expr_seq_elem -> Attribute | Await | ... | Template1 | Template2 | ...
        for parent, children in sorted(preproc.sum_type_constructors.items()):
            assert parent not in rules_mask
            rules_mask[parent] = (offset, offset + len(children))
            offset += len(children)
            all_rules += [(parent, child) for child in children]

        # Rules of the form:
        # FunctionDef
        # -> identifier name, arguments args
        # |  identifier name, arguments args, stmt* body
        # |  identifier name, arguments args, expr* decorator_list
        # |  identifier name, arguments args, expr? returns
        # ...
        # |  identifier name, arguments args, stmt* body, expr* decorator_list, expr returns
        for name, field_presence_infos in sorted(preproc.field_presence_infos.items()):
            assert name not in rules_mask
            rules_mask[name] = (offset, offset + len(field_presence_infos))
            offset += len(field_presence_infos)
            all_rules += [(name, presence) for presence in field_presence_infos]

        # Rules of the form:
        # stmt* -> stmt
        #        | stmt stmt
        #        | stmt stmt stmt
        for seq_type_name, lengths in sorted(preproc.seq_lengths.items()):
            assert seq_type_name not in rules_mask
            rules_mask[seq_type_name] = (offset, offset + len(lengths))
            offset += len(lengths)
            all_rules += [(seq_type_name, i) for i in lengths]

        return all_rules, rules_mask

    def compute_rule_masks(self, grammar_path):

        # paths for head-corner settings
        path_to_preterminals = os.path.join(grammar_path, "preterminals.csv")
        path_to_head_map = os.path.join(grammar_path, "rule_to_head.csv")

        # goal to preterminals
        preterminal_map = {}
        preterminal_masks = {}
        preterminal_debug = {}
        preterminal_types = set()
        with open(path_to_preterminals, 'r') as csv_file:
            elems = csv.reader(csv_file)
            for goal, prets in elems:
                prets = set([s.strip() for s in prets.strip().split(",") if s != ""])
                preterminal_types = preterminal_types.union(prets)

        all_rules = list(self.preproc.all_rules + tuple([("", "**MATCH**")]) + tuple([("", p) for p in preterminal_types]))
        rules_index = {v: idx for idx, v in enumerate(all_rules)}

        with open(path_to_preterminals, 'r') as csv_file:
            elems = csv.reader(csv_file)
            for goal, prets in elems:
                prets = [s.strip() for s in prets.strip().split(",") if s != ""]
                mask_ids = sorted([rules_index[("", p)] for p in prets])
                preterminal_map[goal] = prets
                preterminal_masks[goal] = mask_ids
                preterminal_debug[goal] = set(mask_ids)

        # rule to head
        rule_to_head = {}
        parent_to_rule = collections.defaultdict(list)
        head_to_rule = collections.defaultdict(list)
        parent_to_head = collections.defaultdict(list)
        with open(path_to_head_map, 'r') as csv_file:
            elems = csv.reader(csv_file)
            for rule_type, rule, head in elems:
                rule_to_head[rule] = int(head.strip())-1
                if " -> " in rule:
                    lhs, rhs = rule.split(" -> ")
                else:
                    lhs, rhs = rule.strip().rstrip(" ->"), ""

                if len(rhs):
                    split_rhs = [tuple(elem.strip().split("/")) for elem in rhs.split(",")]
                    parent_to_head[lhs] += [split_rhs[rule_to_head[rule]]]
                    head_to_rule[parent_to_head[lhs][-1][0]] += [(lhs, rule)]
                    parent_to_rule[lhs] += [[elem.strip() for elem in rhs.split(",")]]
                else:
                    parent_to_head[lhs] = []
                    parent_to_rule[lhs] = []

        seq_types = set([lhs for lhs, rhs in rules_index if lhs.endswith("*")])
        for stype in seq_types:
            head = stype.rstrip("*")
            parent_to_head[stype] = [(head, "NULL")]
            head_to_rule[head] += [(stype, stype + " -> " + head + "/NULL")]

        # derive head corners for goal nonterminals
        head_corners = {key: set([v[0] for v in value]) for key, value in parent_to_head.items()}
        def dfs(parent, table, visited=set()):
            for head in table[parent]:
                visited.add(parent)
                if head in visited:
                    table[parent] = table[parent].union(table[head])
                else:
                    dfs(head, table, visited)
                    table[parent] = table[parent].union(table[head])
        for key in preterminal_map:
            dfs(key, head_corners)

        # now get head corner table -- set of grammar rules available
        # given a head and a goal state
        head_corner_table = {}
        hc_debug = {}
        for goal_state in preterminal_masks:
            head_corner_table[goal_state] = collections.defaultdict(set)
            hc_debug[goal_state] = collections.defaultdict(set)
            for head in head_corners[goal_state]:
                for parent, rule in head_to_rule.get(head):
                    if parent in head_corners[goal_state]:
                        head_corner_table[goal_state][head].add(rule)
                        hc_debug[goal_state][head] = set()
                    elif parent == goal_state:
                        head_corner_table[goal_state][head].add(rule)
                        hc_debug[goal_state][head] = set()
        # given a rule, get the set of indices in rule vocab corresponding to it
        for key_1 in head_corner_table:
            for key_2 in head_corner_table[key_1]:
                mask = []
                for rule in head_corner_table[key_1][key_2]:
                    lhs, _ = rule.split(" ->")
                    mask += get_rule_match_indices(lhs, rule, rules_index)
                head_corner_table[key_1][key_2] = sorted(mask)
                hc_debug[key_1][key_2] = set(mask)

        return all_rules, rules_index, preterminal_map, preterminal_masks, preterminal_debug, preterminal_types,\
            head_corners, head_corner_table, hc_debug, parent_to_head, parent_to_rule

    def fetch_head_from_node(self, node):
        parent_type = node["_type"]

        name_to_type = {elem[1]: elem[0] for elem in self.parent_to_head[parent_type]}
        for key in node.keys():
            if key != "_type":
                if key in name_to_type:
                    return key, name_to_type[key]
        return None, []

    def construct_oracle_sequence(self, tree_state, oracle_sequence, goal_type=None):

        goal_was_false = not goal_type

        root_type, node = tree_state

        if isinstance(node, list):  ## you've hit the child of an aggregator, visit each of the children
            if goal_was_false:
                root_type += "*"
                goal_type = root_type

            is_sum_type = root_type[:-1] in self.ast_wrapper.sum_types

            for i, elem in enumerate(node):
                if i == 0:
                    if not is_sum_type:
                        self.construct_oracle_sequence((None, elem), oracle_sequence, goal_type=goal_type)
                    # new here
                    else:
                        self.construct_oracle_sequence((root_type[:-1], elem), oracle_sequence, goal_type=goal_type)
                    # here, check if the parent is a sum type constructor, if so, add an extra expand up
                    #if is_sum_type:    ### commenting out for now
                    #    oracle_sequence.append(ExpandUp(rule=(root_type[:-1], elem["_type"]), goal_type=goal_type))
                    rule = get_rule_string_from_node(root_type, node, self.ast_wrapper)
                    oracle_sequence.append(ExpandUp(rule=rule, goal_type=goal_type))
                else:
                    if not is_sum_type:
                        self.construct_oracle_sequence((None, elem), oracle_sequence, goal_type=None)
                    else:
                        self.construct_oracle_sequence((root_type[:-1], elem), oracle_sequence, goal_type=None)
                    # here, check if the parent is a sum type constructor, if so, replace match with ExpandUP, then **MATCH**
                    #if is_sum_type:
                    #    ## oracle_sequence[-1] = ExpandUp(rule=(root_type[:-1], elem["_type"]), goal_type=goal_type)
                    #    oracle_sequence.append(ExpandUp(rule="**MATCH**", goal_type=root_type[:-1]))

            if goal_was_false:
                oracle_sequence.append(ExpandUp(rule="**MATCH**", goal_type=goal_type))

        elif isinstance(node, dict):  ## you're dealing with a Constructor or a ProductType

            node_type = node["_type"]

            if len(node) == 1:
                try:
                    assert goal_was_false
                except:
                    print(node)
                    raise AssertionError

                oracle_sequence.append(PredictPreterminal(ttype=node_type,
                                                          goal_type=root_type))
                oracle_sequence.append(ExpandUp(rule=(root_type, node_type),
                                                goal_type=root_type))
                oracle_sequence.append(ExpandUp(rule="**MATCH**",
                                                goal_type=root_type))
            else:
                parent_rule = get_rule_string_from_node(root_type, node, self.ast_wrapper)
                if goal_was_false:
                    if parent_rule:
                        goal_type = root_type
                    else:
                        goal_type = node["_type"]

                head_field_name, head_field_type = self.fetch_head_from_node(node)
                self.construct_oracle_sequence((head_field_type, node[head_field_name]), oracle_sequence,
                                               goal_type=goal_type)

                # fetch the right rule form for this dict
                type_info = self.ast_wrapper.singular_types[node_type]
                present = get_field_presence_info(self.ast_wrapper, node, type_info.fields)
                rule = (node['_type'], tuple(present))
                oracle_sequence.append(ExpandUp(rule=rule, goal_type=goal_type))

                leftover_children = [field for (field, p) in zip(self.ast_wrapper.singular_types[node_type].fields,
                                                                 present) if p and field.name != head_field_name]

                for child in leftover_children:
                    self.construct_oracle_sequence((child.type, node[child.name]), oracle_sequence, goal_type=None)

                if parent_rule:
                    oracle_sequence.append(ExpandUp(rule=parent_rule, goal_type=goal_type))

                # see what happens when you change the position of this
                if goal_was_false:
                    if root_type:
                        oracle_sequence.append(ExpandUp(rule="**MATCH**", goal_type=root_type))
                    else:
                        oracle_sequence.append(ExpandUp(rule="**MATCH**", goal_type=node_type))

        else:
            # something going on with singletons that we want to fix
            #if root_type not in ["table", "column"]:
            #    root_type = str(type(node))
            if goal_was_false:
                goal_type = root_type
            # predicting a preterminal, pointing to its value. match if appropriate.
            oracle_sequence.append(PredictPreterminal(ttype=root_type,
                                                      goal_type=goal_type))
            oracle_sequence.append(Point(ttype=root_type,
                                         value=node))
            if goal_was_false:
                oracle_sequence.append(ExpandUp(rule="**MATCH**", goal_type=goal_type))

    def augment_data_with_oracle(self, zipped_data):
        encoder_data, decoder_data = zip(*zipped_data)

        oracle_sequences = []
        for elem in decoder_data:
            oracle_sequence = self.compute_oracle_sequence(elem)
            oracle_sequences.append(oracle_sequence)

        return zip(encoder_data, decoder_data, oracle_sequences)

    def begin_inference(self, desc_enc, example):
        traversal = InferenceTreeTraversal(self, desc_enc, example)
        choices = traversal.step(None)
        return traversal, choices

    def compute_loss(self, enc_input, example, desc_enc, debug):
        mle_loss = self.compute_mle_loss(enc_input, example, desc_enc, debug)

        if self.use_align_loss:
            align_loss = self.compute_align_loss(desc_enc, example[0])
            return mle_loss + align_loss
        return mle_loss

    def init_state(self, enc_input, example, desc_enc):
        self.goals.append(("sql", None))
        self.head_corners = []
        self.operation = self.State.PREDICT_HEAD_CORNER

    def compute_mle_loss(self, enc_input, example, desc_enc, debug):

        _, oracle = example

        # copy this over because pop is destructive
        oracle = oracle[:]

        #print("##########")
        #print(json.dumps(example[0].tree, indent=2))
        #print(oracle)

        traversal = TrainTreeTraversal(self, desc_enc)
        traversal.step(None)
        while oracle:
            action = oracle.pop(0)
            if isinstance(action, PredictPreterminal):
                index = self.rules_index[("", action.ttype)]
                goal_type = action.goal_type
                assert traversal.current_state == TreeTraversal.State.PRETERMINAL_APPLY
                assert traversal.goals[-1].node_type == goal_type
                try:
                    assert index in self.preterminal_debug[goal_type]
                except:
                    # print(action)
                    raise AssertionError
                traversal.step(index)
            elif isinstance(action, ExpandUp):
                hc = traversal.head_corners[-1]
                if action.rule == "**MATCH**":
                    index = self.rules_index[("", action.rule)]
                else:
                    index = self.rules_index[action.rule]
                    assert index in self.hc_debug[hc.goal_type][hc.root_type]
                traversal.step(index)
            else:  # point
                assert traversal.current_state in [TreeTraversal.State.POINTER_APPLY,
                                                   TreeTraversal.State.GEN_TOKEN_APPLY]
                if action.ttype not in ["table", "column"]:
                    # we're doing conventional pointing (which we handle as strings)
                    value = action.value
                    field_value_split = self.preproc.grammar.tokenize_field_value(value) + [
                        vocab.EOS]
                    for value in field_value_split:
                        traversal.step(value)
                else:
                    pointer_map = desc_enc.pointer_maps.get(action.ttype)
                    value = action.value

                    if pointer_map:
                        values = pointer_map[value]
                        traversal.step(values[0], values[1:])
                    else:
                        traversal.step(value)

        loss = torch.sum(torch.stack(tuple(traversal.loss), dim=0), dim=0)

        hc = traversal.head_corners[-1]
        _, converted = traversal.convert_head_corner_to_node_rep(hc)

        # t1 = json.dumps(converted, indent=2, sort_keys=True)
        # t2 = json.dumps(example[0].tree, indent=2, sort_keys=True)

        return loss

    def compute_loss_from_all_ordering(self, enc_input, example, desc_enc, debug):
        def get_permutations(node):
            def traverse_tree(node):
                nonlocal permutations
                if isinstance(node, (list, tuple)):
                    p = itertools.permutations(range(len(node)))
                    permutations.append(list(p))
                    for child in node:
                        traverse_tree(child)
                elif isinstance(node, dict):
                    for node_name in node:
                        traverse_tree(node[node_name])

            permutations = []
            traverse_tree(node)
            return permutations

        def get_perturbed_tree(node, permutation):
            def traverse_tree(node, parent_type, parent_node):
                if isinstance(node, (list, tuple)):
                    nonlocal permutation
                    p_node = [node[i] for i in permutation[0]]
                    parent_node[parent_type] = p_node
                    permutation = permutation[1:]
                    for child in node:
                        traverse_tree(child, None, None)
                elif isinstance(node, dict):
                    for node_name in node:
                        traverse_tree(node[node_name], node_name, node)

            node = copy.deepcopy(node)
            traverse_tree(node, None, None)
            return node

        orig_tree = example.tree
        permutations = get_permutations(orig_tree)
        products = itertools.product(*permutations)
        loss_list = []
        for product in products:
            tree = get_perturbed_tree(orig_tree, product)
            example.tree = tree
            loss = self.compute_mle_loss(enc_input, example, desc_enc)
            loss_list.append(loss)
        example.tree = orig_tree
        loss_v = torch.stack(loss_list, 0)
        return torch.logsumexp(loss_v, 0)

    def _desc_attention(self, prev_state, desc_enc):
        # prev_state shape:
        # - h_n: batch (=1) x emb_size
        # - c_n: batch (=1) x emb_size
        query = prev_state[0]
        if self.attn_type != 'sep':
            return self.desc_attn(query, desc_enc.memory, attn_mask=None)
        else:
            question_context, question_attention_logits = self.question_attn(query, desc_enc.question_memory)
            schema_context, schema_attention_logits = self.schema_attn(query, desc_enc.schema_memory)
            return question_context + schema_context, schema_attention_logits

    def _tensor(self, data, dtype=None):
        return torch.tensor(data, dtype=dtype, device=self._device)

    def _index(self, vocab, word):
        return self._tensor([vocab.index(word)])

    def _update_state(
            self,
            node_type,
            prev_state,
            prev_action_emb,
            prec_h,
            prec_action_emb,
            prec_goal,
            desc_enc):
        # desc_context shape: batch (=1) x emb_size
        desc_context, attention_logits = self._desc_attention(prev_state, desc_enc)
        # node_type_emb shape: batch (=1) x emb_size
        node_type_emb = self.node_type_embedding(
            self._index(self.node_type_vocab, node_type))

        state_input = torch.cat(
            (
                prev_action_emb,  # a_{t-1}: rule_emb_size
                desc_context,  # c_t: enc_recurrent_size
                prec_h,  # s_{p_t}: recurrent_size
                prec_action_emb,  # a_{p_t}: rule_emb_size
                prec_goal,  # recurrent_size (goal node)    CHANGE MADE HERE
                node_type_emb,  # n_{f-t}: node_emb_size
            ),
            dim=-1)
        new_state = self.state_update(
            # state_input shape: batch (=1) x (emb_size * 5)
            state_input, prev_state)
        return new_state, attention_logits

    def apply_rule(
            self,
            node_type,
            prev_state,
            prev_action_emb,
            prec_h,
            prec_action_emb,
            prec_goal,
            desc_enc):

        new_state, attention_logits = self._update_state(
            node_type, prev_state, prev_action_emb, prec_h, prec_action_emb, prec_goal, desc_enc)
        # output shape: batch (=1) x emb_size
        output = new_state[0]
        # rule_logits shape: batch (=1) x num choices
        rule_logits = self.rule_logits(output)

        return output, new_state, rule_logits

    def rule_infer(self, node_type, goal_type, rule_logits, state):
        rule_logprobs = torch.nn.functional.log_softmax(rule_logits, dim=-1)

        if state == TreeTraversal.State.EXPAND_UP_INQUIRE:
            assert goal_type
            rule_ids = self.hc_table[goal_type][node_type]
            if goal_type == node_type:
                rule_ids += [self.rules_index[("", "**MATCH**")]]
                rule_ids = sorted(rule_ids)
        elif state == TreeTraversal.State.PRETERMINAL_INQUIRE:
            rule_ids = self.preterminal_mask[node_type]
        else:
            print("Rule infer should only be evoked for expand up and predict preterminal.")
            raise NotImplementedError

        return list(zip(rule_ids, [rule_logprobs[0, idx] for idx in rule_ids]))


    def gen_token(
            self,
            node_type,
            prev_state,
            prev_action_emb,
            prec_h,
            prec_action_emb,
            goal_h,
            desc_enc):

        new_state, attention_logits = self._update_state(
            node_type, prev_state, prev_action_emb, prec_h, prec_action_emb, goal_h, desc_enc)
        # output shape: batch (=1) x emb_size
        output = new_state[0]

        # gen_logodds shape: batch (=1)
        gen_logodds = self.gen_logodds(output).squeeze(1)

        return new_state, output, gen_logodds

    def gen_token_loss(
            self,
            output,
            gen_logodds,
            token,
            desc_enc):
        # token_idx shape: batch (=1), LongTensor
        token_idx = self._index(self.terminal_vocab, token)
        # action_emb shape: batch (=1) x emb_size
        action_emb = self.terminal_embedding(token_idx)

        # +unk, +in desc: copy
        # +unk, -in desc: gen (an unk token)
        # -unk, +in desc: copy, gen
        # -unk, -in desc: gen
        # gen_logodds shape: batch (=1)
        desc_locs = desc_enc.find_word_occurrences(token)
        if desc_locs:
            # copy: if the token appears in the description at least once
            # copy_loc_logits shape: batch (=1) x desc length
            copy_loc_logits = self.copy_pointer(output, desc_enc.memory)
            copy_logprob = (
                # log p(copy | output)
                # shape: batch (=1)
                    torch.nn.functional.logsigmoid(-gen_logodds) -
                    # xent_loss: -log p(location | output)
                    # TODO: sum the probability of all occurrences
                    # shape: batch (=1)
                    self.xent_loss(copy_loc_logits, self._tensor(desc_locs[0:1])))
        else:
            copy_logprob = None

        # gen: ~(unk & in desc), equivalent to  ~unk | ~in desc
        if token in self.terminal_vocab or copy_logprob is None:
            token_logits = self.terminal_logits(output)
            # shape:
            gen_logprob = (
                # log p(gen | output)
                # shape: batch (=1)
                    torch.nn.functional.logsigmoid(gen_logodds) -
                    # xent_loss: -log p(token | output)
                    # shape: batch (=1)
                    self.xent_loss(token_logits, token_idx))
        else:
            gen_logprob = None

        # loss should be -log p(...), so negate
        loss_piece = -torch.logsumexp(
            maybe_stack([copy_logprob, gen_logprob], dim=1),
            dim=1)
        return loss_piece

    def token_infer(self, output, gen_logodds, desc_enc):
        # Copy tokens
        # log p(copy | output)
        # shape: batch (=1)
        copy_logprob = torch.nn.functional.logsigmoid(-gen_logodds)
        copy_loc_logits = self.copy_pointer(output, desc_enc.memory)
        # log p(loc_i | copy, output)
        # shape: batch (=1) x seq length
        copy_loc_logprobs = torch.nn.functional.log_softmax(copy_loc_logits, dim=-1)
        # log p(loc_i, copy | output)
        copy_loc_logprobs += copy_logprob

        log_prob_by_word = {}
        # accumulate_logprobs is needed because the same word may appear
        # multiple times in desc_enc.words.
        accumulate_logprobs(
            log_prob_by_word,
            zip(desc_enc.words, copy_loc_logprobs.squeeze(0)))

        # Generate tokens
        # log p(~copy | output)
        # shape: batch (=1)
        gen_logprob = torch.nn.functional.logsigmoid(gen_logodds)
        token_logits = self.terminal_logits(output)
        # log p(v | ~copy, output)
        # shape: batch (=1) x vocab size
        token_logprobs = torch.nn.functional.log_softmax(token_logits, dim=-1)
        # log p(v, ~copy| output)
        # shape: batch (=1) x vocab size
        token_logprobs += gen_logprob

        accumulate_logprobs(
            log_prob_by_word,
            ((self.terminal_vocab[idx], token_logprobs[0, idx]) for idx in range(token_logprobs.shape[1])))

        return list(log_prob_by_word.items())

    def compute_pointer(
            self,
            node_type,
            prev_state,
            prev_action_emb,
            parent_h,
            parent_action_emb,
            desc_enc):
        new_state, attention_logits = self._update_state(
            node_type, prev_state, prev_action_emb, parent_h, parent_action_emb, desc_enc)
        # output shape: batch (=1) x emb_size
        output = new_state[0]
        # pointer_logits shape: batch (=1) x num choices
        pointer_logits = self.pointers[node_type](
            output, desc_enc.pointer_memories[node_type])

        return output, new_state, pointer_logits, attention_logits

    def pointer_infer(self, node_type, logits):
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        return list(zip(
            # TODO batching
            range(logits.shape[1]),
            logprobs[0]))