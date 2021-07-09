import enum

import attr
import pyrsistent

from ratsql.models.head_corner import decoder
from ratsql.utils import vocab


@attr.s
class TreeState:
    node = attr.ib()
    parent_field_type = attr.ib()


def get_head_of_grammar_rule(named_children, parent_to_head):
	for i, (child_type, child_name) in enumerate(named_children):
		for head_type,  head_name in parent_to_head:
			if child_type == head_type:
				return i
	raise IndexError


def convert_primitive(primitive):
    if isinstance(primitive, int):
        return primitive
    elif isinstance(primitive, str):
        if primitive == 'False':
            return False
        elif primitive == 'True':
            return True
        else:
            print("The terminal value " + primitive + " is unaccounted for in hc converter.")
            raise NotImplementedError
    else:
        raise NotImplementedError

class TreeTraversal:
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

    @attr.s(frozen=True)
    class GoalItem:
        node_type = attr.ib()
        siblings = attr.ib()  # the siblings that got initialized with this goal (a list of head corners)
        position = attr.ib()  # the position of this goal in the sibling list
        prec_action_emb = attr.ib()  # the rule that created this goal
        prec_h = attr.ib()  #  the hidden state before this goal was created
        goal_h = attr.ib()
        preterminal_done = attr.ib()

        #def to_str(self):
        #   return f"node_type: {self.node_type}, prec_field_name: {self.prec}>"

    @attr.s(frozen=True)
    class HeadCornerItem:
        root_type = attr.ib()
        children = attr.ib()
        prec_action_emb = attr.ib()  # the rule that created this goal
        prec_h = attr.ib()  # the hidden state before this goal was created
        goal_h = attr.ib()
        name = attr.ib()

        #def to_str(self):
        #    return f"<state: {self.state}, node_type: {self.node_type}, parent_field_name: {self.parent_field_name}>"

    @attr.s(frozen=True)
    class TerminalItem:
        value = attr.ib()

    class State(enum.Enum):
        PRETERMINAL_INQUIRE = 0
        PRETERMINAL_APPLY = 1
        EXPAND_UP_INQUIRE = 2
        EXPAND_UP_APPLY = 3
        POINTER_INQUIRE = 4
        POINTER_APPLY = 5
        GEN_TOKEN_INQUIRE = 6
        GEN_TOKEN_APPLY = 7
        DONE = 8

    def __init__(self, model, desc_enc):
        if model is None:
            return

        self.model = model
        self.desc_enc = desc_enc

        model.state_update.set_dropout_masks(batch_size=1)
        self.recurrent_state = decoder.lstm_init(
            model._device, None, self.model.recurrent_size, 1
        )
        self.prev_action_emb = model.zero_rule_emb

        root_type = model.preproc.grammar.root_type

        self.current_state = TreeTraversal.State.PRETERMINAL_INQUIRE
        self.goals = pyrsistent.pvector()
        self.head_corners = pyrsistent.pvector()

        init_goal = TreeTraversal.GoalItem(
            node_type=root_type,
            siblings=[],
            position=None,
            prec_action_emb=self.model.zero_rule_emb,
            prec_h=self.model.zero_recurrent_emb,
            goal_h=self.model.zero_recurrent_emb,
            preterminal_done=False,
        )
        self.goals = self.goals.append(init_goal)

        self.update_prev_action_emb = TreeTraversal._update_prev_action_emb_apply_rule

    def clone(self):
        other = self.__class__(None, None)
        other.model = self.model
        other.desc_enc = self.desc_enc
        other.recurrent_state = self.recurrent_state
        other.prev_action_emb = self.prev_action_emb
        other.queue = self.queue
        other.cur_item = self.cur_item
        other.next_item_id = self.next_item_id
        other.actions = self.actions
        other.update_prev_action_emb = self.update_prev_action_emb
        return other

    def step(self, last_choice, extra_choice_info=None, attention_offset=None):
        while True:
            self.update_using_last_choice(
                last_choice, extra_choice_info, attention_offset
            )

            handler_name = TreeTraversal.Handler.handlers[self.current_state]
            handler = getattr(self, handler_name)
            choices, continued = handler(last_choice)
            if continued:
                last_choice = choices
                continue
            else:
                return choices

    def update_using_last_choice(
            self, last_choice, extra_choice_info, attention_offset
    ):
        if last_choice is None:
            return
        # print("evoke self.update_prev_action_emb")
        self.update_prev_action_emb(self, last_choice, extra_choice_info)

    @classmethod
    def _update_prev_action_emb_apply_rule(cls, self, last_choice, extra_choice_info):
        # print("_update_prev_action_emb_apply_rule")
        rule_idx = self.model._tensor([last_choice])
        self.prev_action_emb = self.model.rule_embedding(rule_idx)

    @classmethod
    def _update_prev_action_emb_gen_token(cls, self, last_choice, extra_choice_info):
        # token_idx shape: batch (=1), LongTensor
        # print("update_prev_action_emb_gen_token")
        token_idx = self.model._index(self.model.terminal_vocab, last_choice)
        # action_emb shape: batch (=1) x emb_size
        self.prev_action_emb = self.model.terminal_embedding(token_idx)

    @classmethod
    def _update_prev_action_emb_pointer(cls, self, last_choice, extra_choice_info):
        # print("update_prev_action_emb_pointer")
        node_type = self.head_corners[-1].root_type
        self.prev_action_emb = self.model.pointer_action_emb_proj[
            node_type
        ](self.desc_enc.pointer_memories[node_type][:, last_choice])

    def pop(self):
        if self.queue:
            self.cur_item = self.queue[-1]
            self.queue = self.queue.delete(-1)
            return True
        return False

    @Handler.register_handler(State.PRETERMINAL_INQUIRE)
    def process_preterminal_inquire(self, last_choice):
        # 1. PredictPreterminal, like sql >> column
        # just treat these as special kinds of rules
        current_goal = self.goals[-1]

        output, self.recurrent_state, rule_logits = self.model.apply_rule(
            current_goal.node_type,
            self.recurrent_state,
            self.prev_action_emb,
            current_goal.prec_h,
            current_goal.prec_action_emb,
            current_goal.goal_h,
            self.desc_enc
        )

        self.goals = self.goals.set(-1, attr.evolve(self.goals[-1], prec_h=output))
        self.current_state = TreeTraversal.State.PRETERMINAL_APPLY

        choices = self.rule_choice(current_goal.node_type, rule_logits)

        self.update_prev_action_emb = (
            TreeTraversal._update_prev_action_emb_apply_rule
        )

        return choices, False

    @Handler.register_handler(State.PRETERMINAL_APPLY)
    def process_preterminal_apply(self, last_choice):
        _, node_type = self.model.all_rules[last_choice]

        new_hc = TreeTraversal.HeadCornerItem(
            root_type=node_type,
            children=[],
            prec_action_emb=self.prev_action_emb,
            prec_h=self.goals[-1].prec_h,
            goal_h=self.goals[-1].prec_h,
            name=""
        )

        self.head_corners = self.head_corners.append(new_hc)
        self.goals = self.goals.set(-1, attr.evolve(self.goals[-1], preterminal_done=True))

        if node_type in self.model.preproc.grammar.pointers:
            self.current_state = TreeTraversal.State.POINTER_INQUIRE
        elif node_type in self.model.ast_wrapper.primitive_types:
            self.current_state = TreeTraversal.State.GEN_TOKEN_INQUIRE
        else:
            self.current_state = TreeTraversal.State.EXPAND_UP_INQUIRE

        last_choice = None
        return last_choice, True

    @Handler.register_handler(State.POINTER_INQUIRE)
    def process_pointer_inquire(self, last_choice):
        # a. Ask which one to choose
        # print("process_pointer_inquire")
        head_corner = self.head_corners[-1]

        output, self.recurrent_state, logits, attention_logits = self.model.compute_pointer_with_align(
            head_corner.root_type,
            self.recurrent_state,
            self.prev_action_emb,
            head_corner.prec_h,
            head_corner.prec_action_emb,
            head_corner.goal_h,
            self.desc_enc
        )

        self.head_corners = self.head_corners.set(-1, attr.evolve(
            head_corner, prec_h=output)
        )
        self.current_state = TreeTraversal.State.POINTER_APPLY

        self.update_prev_action_emb = (
            TreeTraversal._update_prev_action_emb_pointer
        )
        choices = self.pointer_choice(
            head_corner.root_type, logits, attention_logits
        )
        return choices, False

    @Handler.register_handler(State.POINTER_APPLY)
    def process_pointer_apply(self, last_choice):
        # create a dummy hc for the pointer value
        terminal = TreeTraversal.TerminalItem(value=last_choice)

        self.head_corners = self.head_corners.set(-1, attr.evolve(self.head_corners[-1], children=[terminal]))

        self.current_state = TreeTraversal.State.EXPAND_UP_INQUIRE

        return None, True

    @Handler.register_handler(State.GEN_TOKEN_INQUIRE)
    def process_gen_token_inquire(self, last_choice):

        head_corner = self.head_corners[-1]

        self.recurrent_state, output, gen_logodds = self.model.gen_token(
            head_corner.root_type,
            self.recurrent_state,
            self.prev_action_emb,
            head_corner.prec_h,
            head_corner.prec_action_emb,
            head_corner.goal_h,
            self.desc_enc,
        )

        self.update_prev_action_emb = (
            TreeTraversal._update_prev_action_emb_gen_token
        )
        self.head_corners = self.head_corners.set(-1, attr.evolve(
            head_corner, prec_h=output)
        )

        self.current_state = TreeTraversal.State.GEN_TOKEN_APPLY

        choices = self.token_choice(output, gen_logodds)

        return choices, False

    @Handler.register_handler(State.GEN_TOKEN_APPLY)
    def process_gen_token_apply(self, last_choice):

        if last_choice == vocab.EOS:
            self.current_state = TreeTraversal.State.EXPAND_UP_INQUIRE
            return None, True

        terminal = TreeTraversal.TerminalItem(value=last_choice)
        head_corner_children = self.head_corners[-1].children

        self.head_corners = self.head_corners.set(-1, attr.evolve(self.head_corners[-1],
                                                                  children=head_corner_children + [terminal]))

        self.current_state = TreeTraversal.State.GEN_TOKEN_INQUIRE

        return None, True

    @Handler.register_handler(State.EXPAND_UP_INQUIRE)
    def process_expand_up_inquire(self, last_choice):
        # 1. ExpandUp, like column << col_unit -> agg_type, column, singleton
        head_corner = self.head_corners[-1]

        output, self.recurrent_state, rule_logits = self.model.apply_rule(
            head_corner.root_type,
            self.recurrent_state,
            self.prev_action_emb,
            head_corner.prec_h,
            head_corner.prec_action_emb,
            head_corner.goal_h,
            self.desc_enc
        )

        self.head_corners = self.head_corners.set(-1, attr.evolve(self.head_corners[-1], prec_h=output))
        self.current_state = TreeTraversal.State.EXPAND_UP_APPLY

        choices = self.rule_choice(head_corner.root_type, rule_logits)

        self.update_prev_action_emb = (
            TreeTraversal._update_prev_action_emb_apply_rule
        )

        return choices, False

    @Handler.register_handler(State.EXPAND_UP_APPLY)
    def process_expand_up_apply(self, last_choice):
        lhs, rhs = self.model.all_rules[last_choice]

        if rhs == "**MATCH**":

            goal = self.goals[-1]
            self.goals = self.goals.delete(-1)

            if not len(self.goals):
                # you've popped the last goal, so the only head corner is the final structure
                self.current_state = TreeTraversal.State.DONE
                return None, False

            else:
                # this head corner should become the child of a pre-existing head corner
                head_corner = self.head_corners[-1]
                name = goal.siblings[goal.position].name
                goal.siblings[goal.position] = attr.evolve(head_corner, name=name)
                self.head_corners = self.head_corners.delete(-1)

                if self.goals[-1].preterminal_done:
                    self.current_state = TreeTraversal.State.EXPAND_UP_INQUIRE
                else:
                    self.current_state = TreeTraversal.State.PRETERMINAL_INQUIRE

        else:
            head_corner = self.head_corners[-1]
            if isinstance(rhs, str):
                new_root = TreeTraversal.HeadCornerItem(root_type=lhs,
                                                        children=[head_corner],
                                                        prec_action_emb=self.prev_action_emb,
                                                        prec_h=head_corner.prec_h,
                                                        goal_h=head_corner.goal_h,
                                                        name="")

                self.head_corners = self.head_corners.set(-1, new_root)

                self.current_state = TreeTraversal.State.EXPAND_UP_INQUIRE

            elif type(rhs) in [tuple, list]:

                parent_rules = self.model.parent_to_rule[lhs]
                children_names = parent_rules[0]
                kept_children = [child.split("/") for i, child in enumerate(children_names) if rhs[i]]
                head_index = get_head_of_grammar_rule(kept_children, self.model.parent_to_head[lhs])
                children = []
                for child_type, child_name in kept_children:
                    child = TreeTraversal.HeadCornerItem(root_type=child_type,
                                                         children=[],
                                                         prec_action_emb=self.prev_action_emb,
                                                         prec_h=head_corner.prec_h,
                                                         goal_h=head_corner.goal_h,
                                                         name=child_name)
                    children.append(child)

                head_type, head_name = children[head_index].root_type, children[head_index].name
                assert head_corner.root_type == head_type
                children[head_index] = attr.evolve(head_corner, name=head_name)

                new_root = TreeTraversal.HeadCornerItem(root_type=lhs,
                                                        children=children,
                                                        prec_action_emb=self.prev_action_emb,
                                                        prec_h=head_corner.prec_h,
                                                        goal_h=head_corner.goal_h,
                                                        name="")

                self.head_corners = self.head_corners.set(-1, new_root)

                for i in reversed(range(len(children))):
                    if i != head_index:     # all the newly created children that need children of their own
                        goal = TreeTraversal.GoalItem(node_type=children[i].root_type,
                                                      siblings=children,
                                                      position=i,
                                                      prec_action_emb=self.prev_action_emb,
                                                      prec_h=head_corner.prec_h,
                                                      goal_h=head_corner.goal_h,
                                                      preterminal_done=False)
                        self.goals = self.goals.append(goal)

                if len(children) > 1:
                    self.current_state = TreeTraversal.State.PRETERMINAL_INQUIRE
                else:
                    self.current_state = TreeTraversal.State.EXPAND_UP_INQUIRE

            elif isinstance(rhs, int):
                # children don't have names here
                child_type = lhs.rstrip("*")
                head_index = 0
                children = []
                for i in range(rhs):
                    child = TreeTraversal.HeadCornerItem(root_type=child_type,
                                                         children=[],
                                                         prec_action_emb=self.prev_action_emb,
                                                         prec_h=head_corner.prec_h,
                                                         goal_h=head_corner.goal_h,
                                                         name="")
                    children.append(child)

                children[head_index] = head_corner

                new_root = TreeTraversal.HeadCornerItem(root_type=lhs,
                                                        children=children,
                                                        prec_action_emb=self.prev_action_emb,
                                                        prec_h=head_corner.prec_h,
                                                        goal_h=head_corner.goal_h,
                                                        name="")

                self.head_corners = self.head_corners.set(-1, new_root)

                for i in reversed(range(len(children))):
                    if i != head_index:
                        goal = TreeTraversal.GoalItem(node_type=children[i].root_type,
                                                      siblings=children,
                                                      position=i,
                                                      prec_action_emb=self.prev_action_emb,
                                                      prec_h=head_corner.prec_h,
                                                      goal_h=head_corner.goal_h,
                                                      preterminal_done=False)

                        self.goals = self.goals.append(goal)

                if len(children) > 1:
                    self.current_state = TreeTraversal.State.PRETERMINAL_INQUIRE
                else:
                    self.current_state = TreeTraversal.State.EXPAND_UP_INQUIRE

            else:
                raise NotImplementedError

        return None, True

    def rule_choice(self, node_type, rule_logits):
        raise NotImplementedError

    def token_choice(self, output, gen_logodds):
        raise NotImplementedError

    def pointer_choice(self, node_type, logits, attention_logits):
        raise NotImplementedError

    def convert_head_corner_to_node_rep(self, hc):
        if isinstance(hc, TreeTraversal.TerminalItem):
            return convert_primitive(hc.value)
        else:
            node_type = hc.root_type
            if node_type.endswith('*'):
                return [self.convert_head_corner_to_node_rep(child) for child in hc.children]
            elif node_type in self.model.ast_wrapper.sum_types:
                return self.convert_head_corner_to_node_rep(hc.children[0])
            elif node_type in self.model.ast_wrapper.primitive_types:
                return self.convert_head_corner_to_node_rep(hc.children[0])
            else:
                contribution = {"_type": node_type}
                for child in hc.children:
                    name = child.name
                    contribution.update({name: self.convert_head_corner_to_node_rep(child)})
                return contribution

