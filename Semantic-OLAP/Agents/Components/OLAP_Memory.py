import json
import pandas as pd
from Utils.jsonfy_result import jsonfy_llm_response

PROMPT_GET_NODE = """
You are a query semantics analysis assistant. Please compare the data sets described by the "current query" and the "historical query", and output only the specified keyword indicating their relationship.

Available relationship keywords:
- contain  : The data set of the current query strictly contains that of the historical query.  
  - In terms of filtering: The current query has looser conditions or removes some constraints from the historical query. All results returned by the historical query would also be included in the current query, though the current one may return more.

- subset   : The data set of the current query is strictly contained within that of the historical query.  
  - In terms of filtering: The current query has stricter conditions or adds constraints on top of the historical query. All data in the current query also satisfies the historical query, but not vice versa.

- equal    : The two queries describe exactly the same data set.  
  - In terms of filtering: Both queries use the same fields, conditions, and values—even if worded differently, the logical result set is the same.

- intersect: The two data sets definitely have overlap, but neither contains the other.  
  - In terms of filtering: There is partial overlap in fields or values, but the overall conditions don't form a containment relationship. The queries are known to return some common data.

- no_relation: The two data sets definitely do not overlap, or it's impossible to determine whether there is any intersection.  
  - In terms of filtering: The queries contain mutually exclusive conditions (e.g., non-overlapping value ranges for the same field), or there is insufficient information to determine overlap.

Judgment Guidelines:
1. If one query adds conditions or narrows value ranges on top of the other, then its data set is a subset of the other.  
   - If the current query is stricter, it is a subset of the historical query.  
   - If the current query is looser, it contains the historical query.

2. Use intersect only when it's clear the two queries share data but are not subsets. If overlap cannot be determined, use no_relation.

Output format:  
Return a single-line JSON only, with no extra text:  
{"reason": "Your reasoning — list and compare filter conditions of both queries before drawing a conclusion", "relation": "<contain|subset|equal|intersect|no_relation>"}

---

[Current Query]  
%s

[Historical Query]  
%s

Please analyze and determine the relationship:
"""



def parse_relation(ret_str):
    try:
        ret = jsonfy_llm_response(ret_str)
        relation = ret["relation"]
    except:
        relation = "No_relation"

    if ('contain' in relation.lower()):
        return "Contain"
    elif ('subset' in relation.lower()):
        return "Subset"
    elif ('equal' in relation.lower()):
        return "Equal"
    elif ('intersect' in relation.lower()):
        return "Intersect"
    else:
        return "No_relation"

from typing import Dict, List, Optional, Set, Tuple


class Granularity_View:
    def __init__(self, df, desc):
        self.df = df
        self.desc = desc
        self.view = None
        self.plan = []

    def set_view(self, df):
        self.view = df


class Col_Views:

    def __init__(self, root_name: str, root_view: Granularity_View, desc: str, plan):
        self.root: str = root_name
        self._nodes: Dict[str, Granularity_View] = {}
        self.desc = desc
        self.add_node(root_name, root_view)
        if(plan is not None):
            self.plan = [plan]
        else:
            self.plan = None

    def add_node(self, name: str, gv: Granularity_View):
        self._nodes[name] = gv

    def get_view(self, name: str) -> Optional[Granularity_View]:
        return self._nodes.get(name)


    def get_all_nodes(self) -> List[str]:
        return list(self._nodes.keys())

    def get_all_edges(self) -> List[tuple]:
        return []


class Cube_node:
    def __init__(self, id, query, docs_df):
        self.id = id
        self.query = query
        self.docs_df = docs_df.reset_index(drop=True)
        self.docs_df["OLAP_ID"] = range(len(self.docs_df))
        self.col_head = {
            col: Col_Views(root_name=col, root_view=Granularity_View(
                self.docs_df[["OLAP_ID", col]], desc=col), desc=col, plan = None)
            for col in self.docs_df.columns
        }
        self.plan = None


    def add_column(self, name, data, desc, plan):
        self.docs_df[name] = data[name]
        root_view = Granularity_View(data, f"{name} : {desc}")
        self.col_head[name] = Col_Views(root_name=name, root_view=root_view, desc = desc, plan = plan)

    def get_dimension(self, params):
        lines = list(self.col_head.keys())
        return json.dumps(lines)

    def get_granularity(self, params):
        dimension = params["dimension"]
        if(dimension not in self.col_head.keys()):
            return f"Dimension '{dimension}' does not exist."
        granularity = self.col_head[dimension].get_all_nodes()
        return f"Dimension '{dimension}' has the following granularity: {granularity}"


class OLAP_Memory:
    def __init__(self, llm):
        self.nodes = {}
        self.edges = {}
        self.llm = llm

    def init(self, docs_df):
        self.add_node("All Data", docs_df)
        self.root = 0

    def get_node(self, node_id):
        return self.nodes[node_id]

    def add_node(self, query, docs):
        id = len(self.nodes)
        new_node = Cube_node(id, query, docs)
        self.nodes[id] = new_node
        self.edges[id] = []
        return id

    def add_edge(self, from_id, to_id):
        self.edges[from_id].append(to_id)

    def get_relation(self, query_now, query_node):
        prompt = PROMPT_GET_NODE % (query_now, query_node)
        response_str = self.llm.predict(prompt)
        result = parse_relation(response_str)
        return result

    def get_current_node(self, query):
        choosed_nodes = []
        nodes_queue = [self.root]
        while (nodes_queue):
            node_now = nodes_queue.pop(0)
            if (len(self.edges[node_now]) == 0):
                choosed_nodes.append(node_now)
                continue
            contain_childs = []
            for child in self.edges[node_now]:
                child_node = self.nodes[child]
                relation = self.get_relation(query, child_node.query)
                if (relation == "Equal"):
                    return [child], "Equal"
                elif (relation == "Subset"):
                    contain_childs.append(child)
            if (contain_childs == []):
                choosed_nodes.append(node_now)
            else:
                nodes_queue.extend(contain_childs)

        return choosed_nodes, "Subset"


