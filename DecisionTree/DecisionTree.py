import collections
import math


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


# Node object
class Node:
    def __init__(self, data, labels, base=2, max_depth=None,
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
        self.feat_value = feat_value
        self.node_type = node_type
        self.best_split_feature(node_type)

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
            _new_node = self.__class__(_new_node_data, _new_node_labels,
                                       base=self._base, max_depth=self._max_depth, parent=self,
                                       depth=self._depth + 1, is_root=False, feat_value=feat,
                                       node_type=self.node_type)
            self.children[feat] = _new_node
            _new_node.fit()

    def stop(self, eps=1e-8):
        if (self._data.shape[1] == 0 or (self.entropy is not None and self.entropy <= eps)
            or (self._max_depth is not None and self._depth >= self._max_depth)):
            # print("Leaf node at ", self._data.shape, self.entropy, set(self._labels))
            return True
        return False

    def fit(self, eps=1e-8):
        if self.stop(eps):
            return
        _max_feature = self._split_feature
        self._generate_children(_max_feature)

    def view(self, indent=4):
        if self.parent is not None:
            print(' ' * indent * self._depth, self.parent._split_feature, self.feat_value)
        else:
            print(' ' * indent * self._depth, "Root")
        for _node in self.children.values():
            _node.view()

# Tree

