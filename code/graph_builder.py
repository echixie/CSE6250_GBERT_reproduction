from utils import Voc
from constants import ROOT_NODE_MED, ROOT_NODE_DIAG, CODE_SPLIT_INDEX, DIAG_LEAF_NODES

"""
:param nodes_list: list of nodes of a code. eg: ['abcd', 'abc', 'ab', 'c', 'root_node']
:param voc_map: map between node and its index 
:param is_to_children: if true, build edges from ancestor to children. Otherwise, build edges from leaf to ancestor
"""
def build_tree_edges(nodes_list, voc_map, is_to_children):
	node_edges = []

	for nodes in nodes_list:
		node_index = []
		# find list of node's index
		for node in nodes:
			node_index.append(voc_map.word2idx[node])

		if is_to_children:
			# ancestor -> children
			for i in range(len(node_index) - 1):
				node_edges.append((node_index[i+1], node_index[i]))
		else :
			# bottom leaf node -> ancestor
			for i in range(1, len(node_index)):
				node_edges.append((node_index[0], node_index[i]))

	upper_nodes = []
	lower_nodes = []
	for edge in list(set(node_edges)):
		upper_nodes.append(edge[0])
		lower_nodes.append(edge[1])

	return [upper_nodes, lower_nodes]

"""
:param codes: codes representing medication records. eg: ['FADG', 'ADFAD']
"""
def build_tree_med(codes):
	nodes_list = []
	voc_map = Voc()
	for code in codes:
		nodes = []
		nodes.append(code)
		for i in CODE_SPLIT_INDEX:
			nodes.append(code[:i])
		nodes.append(ROOT_NODE_MED)

		voc_map.add_sentence(nodes)
		nodes_list.append(nodes)
	return nodes_list, voc_map

"""
:param codes: codes representing diagnosis records. eg: ['V123', 'E3123']
"""
def build_tree_diag(codes):
	# build nodes map for each code
	code_map = {}

	for code_range in DIAG_LEAF_NODES:
		split_codes = code_range.split('-')
		length = len(split_codes)
		if length == 1:
			code_map[code_range] = code_range
		else:
			if(code_range[0].isdigit()):
				for index in range(int(split_codes[0]), int(split_codes[1]) + 1):
					# i.e: 123
					code_map["%03d" % index] = code_range
			else:
				for index in range(int(split_codes[0][1:]), int(split_codes[1][1:]) + 1):
					# i.e: V34, E102
					key = "V%02d" % index if code_range[0] == 'V' else "E%03d" % index
					code_map[key] = code_range

	# build diagnosis tree
	nodes_list = []
	voc_map = Voc()
	for code in codes:
		nodes = []
		# attach each level tree node to the list
		nodes.append(code)
		trim_code = code[:4] if code[0] == 'E' else code[:3]
		nodes.append(trim_code)
		nodes.append(code_map[trim_code])
		nodes.append(ROOT_NODE_DIAG)

		voc_map.add_sentence(nodes)
		nodes_list.append(nodes)
	return nodes_list, voc_map

def main():
    Wm = WordsMap()
    strings = ['FSAD','FSAD']
    aa = 'E90'
    res, voc = build_tree_med(strings)
    # res, vod = build_tree_diag(strings)
    # print(res)
    # print(voc.word_to_index)
    # print(aa[0].isdigit())
    aa = build_tree_edges(res, voc, False)
    print(aa)

if __name__ == "__main__":
    main()