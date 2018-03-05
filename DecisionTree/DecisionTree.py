import collections
import math
import numpy as np

# Utils: tools to calculate entropies
class Utils:
    def __init__(self, data, labels, base=2):
        self._data = data
        self._counters = collections.Counter(labels)
        self._labels = labels
        self._base = base

    def entropy(self, labels=None, eps=1e-12):
        _len = len(self._labels)
        if labels is None:
            labels = [val for val in self._counters.values()]
        return max([eps, - np.sum([p / _len * math.log(p / _len, self._base) for p in labels])])

    def cond_entropy(self, idx):
        data = self._data[idx]
        values = set(data)
        result = 0
        for val in values:
            index = data == val
            labels = self._labels[index]
            result += Utils(data[index], labels).entropy() * len(labels) / len(data)
        return result

    def info_gain(self, idx):
        _cond_ent = self.cond_entropy(idx)
        _gain = self.entropy() - _cond_ent
        return _gain, _cond_ent

    def feature_entropy(self, idx, eps=1e-12):
        _feat_values = self._data[idx]
        _len = len(_feat_values)
        _feat_labels = Counter(_feat_values).values()
        return max([eps, - np.sum([p / _len * math.log(p / _len, self._base) for p in _feat_labels])])

    def info_gain_rate(self, idx):
        _feat_ent = self.feature_entropy(idx)
        return self.info_gain(idx)[0] / _feat_ent


# Node object: main class to realize a classification decision tree
class Node:
    def __init__(self, data, labels, tree=None, base=2, max_depth=None,
                 depth=0, parent=None, is_root=True, feat_value=None,
                 node_type="ID3"):
        self._data = data
        self._labels = labels
        self._features = data.columns
        self._base = base
        self._utils = Utils(self._data, self._labels, self._base)
        self._max_depth = max_depth
        self._depth = depth
        self.entropy = self._utils.entropy()
        self.is_root = is_root
        self.parent = parent
        self.leafs = {}
        self.children = {}
        self.feat_value = feat_value  # to record the value of split feature
        self.node_type = node_type
        self.best_split_feature(node_type)
        self.class_result = None  # save the node class if the node is a leaf
        self.tree = tree
        self.pruned = False
        if self.tree is not None:
            tree.nodes.append(self)  # add this node into the tree object

    @property
    def key(self):
        return self._depth, self._split_feature, id(self)

    @property
    def layers(self):
        if self.class_result is not None:
            return 1
        return 1 + max([_child.layers for _child in self.children.values()])

    def best_split_feature(self, node_type="ID3"):
        max_gain = - np.infty
        feat = ""
        for feature in self._features:
            new_gain = self._utils.info_gain_rate(feature) if node_type == "C4.5" else self._utils.info_gain(feature)[0]
            if max_gain < new_gain:
                max_gain = new_gain
                feat = feature
        self._split_feature = feat
        self._info_gain = max_gain
        return feat, max_gain

    def _generate_children(self, feat_name):
        feat_values = self._data[feat_name]
        _new_data = self._data.drop(feat_name, axis=1)
        for feat in set(feat_values):
            _idx = feat_values == feat
            _new_node_data = _new_data[_idx]
            _new_node_labels = self._labels[_idx]
            _new_node = self.__class__(_new_node_data, _new_node_labels, tree=self.tree,
                                       base=self._base, max_depth=self._max_depth, parent=self,
                                       depth=self._depth + 1, is_root=False, feat_value=feat,
                                       node_type=self.node_type)
            self.children[feat] = _new_node
            _new_node.fit()

    def stop(self, eps=1e-8):
        if (self._data.shape[1] == 0 or (self.entropy is not None and self.entropy <= eps)
            or (self._max_depth is not None and self._depth >= self._max_depth)
            or len(set(self._labels)) == 1):
            self.class_result = self.get_class()
            # print("Leaf node at ", self._data.shape, self.entropy, set(self._labels))
            _parent = self
            while _parent is not None:
                _parent.leafs[self.key] = self
                _parent = _parent.parent
            return True
        return False

    def fit(self, eps=1e-8):
        if self.stop(eps):
            return
        _max_feature = self._split_feature
        self._generate_children(_max_feature)

    def get_class(self):  # if the node is leaf, then return the classification result of this node
        _counter = Counter(self._labels)
        return max(_counter.keys(), key=lambda key: _counter[key])

    def view(self, indent=4):
        if self.parent is not None:
            if self.class_result is None:
                print(' ' * indent * self._depth, self.parent._split_feature, self.feat_value)
            else:
                print(' ' * indent * self._depth, self.parent._split_feature, self.feat_value, "class: ",
                      self.class_result)
        else:
            print(' ' * indent * self._depth, "Root")
        for _node in self.children.values():
            _node.view()

    def predict_one_sample(self, x):
        if self.class_result is not None:
            return self.class_result
        else:
            try:
                return self.children[x[self._split_feature][0]].predict_one_sample(x)
            except KeyError:
                return self.get_class()

    def predict(self, data):
        if self.class_result is not None:
            if self.is_root:
                return [self.class_result] * len(data)
            return self.class_result
        return [self.predict_one_sample(data.loc[i:i]) for i in range(len(data))]

    def prune_node(self):  # Prune this node: 1. give a category for this node; 2. delete all leafs of this node\
        # 3. empty the children of this node
        if self.class_result is None:
            self.class_result = self.get_class()
        _leafs_to_pop = [key for key in self.leafs.keys()]
        for key in _leafs_to_pop:
            if not key == self.key:
                self._data = self._data.append(self.leafs[key]._data)
                self._labels = self._labels.append(self.leafs[key]._labels)
        _parent = self
        self._utils = Utils(self._data, self._labels, self._base)
        while _parent is not None:
            for key in _leafs_to_pop:
                _parent.leafs.pop(key)
            _parent.leafs[self.key] = self
            _parent = _parent.parent
        self.children = {}
        self.pruned = True

    # Attention ! This is not the entropy we used to build nodes. It's the loss function to prune the tree !
    def loss_if_pruned(self, eps=1e-12):
        _leafs_to_pop = [key for key in self.leafs.keys()]
        _data = self._data
        _labels = self._labels
        for key in _leafs_to_pop:
            if not key == self.key:
                _labels = _labels.append(self.leafs[key]._labels)
        _label_counter = Counter(_labels)
        _labels_ent = [val for val in _label_counter.values()]
        _len = len(_labels)
        return max([eps, - np.sum([p * math.log(p / _len, self._base) for p in _labels_ent])])


# Tree: this class is used for controlling globally nodes, thus for tree pruning
class Tree:
    def __init__(self, data, labels, max_depth=None, node_type="ID3"):
        self.nodes = []
        self._max_depth = max_depth
        self.node_type = node_type
        self.root = Node(data, labels, tree=self, max_depth=max_depth, node_type=self.node_type)

    @property
    def depth(self):
        return self.root.layers

    def fit(self, eps=1e-8):
        self.root.fit(eps)

    def predict_one_sample(self, x):
        return self.root.predict_one_sample(x)

    def predict(self, data):
        return self.root.predict(data)

    def prune(self, alpha=100):
        if self.depth <= 2:
            return
        _tmp_nodes = sorted(np.array([node for node in self.nodes if not node.is_root and not node.class_result]),
                            key=lambda node: node.layers)
        _old_loss = np.array([np.sum([leaf._utils.entropy() * len(leaf._labels) for leaf in node.leafs.values()])
                              + alpha * len(node.leafs) for node in _tmp_nodes])
        _new_loss = [node.loss_if_pruned() + alpha for node in _tmp_nodes]
        _idx = (_old_loss - _new_loss) > 0
        arg = np.argmax(_idx)
        if _idx[arg]:
            # print("Node to prune", _tmp_nodes[arg].key, "Entropy", _new_loss[arg], _old_loss[arg])
            _tmp_nodes[arg].prune_node()
            for i in range(len(self.nodes) - 1, -1, -1):
                if self.nodes[i].pruned:
                    self.nodes.pop(i)
            self.prune(alpha)