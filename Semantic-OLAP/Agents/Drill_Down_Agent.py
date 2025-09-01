import json
from Utils.send_logs import debug_log
from Agents.Components.Operaters import sem_map
from Utils.jsonfy_result import jsonfy_llm_response


PROMPT_DRILLDOWN_DIMENSION = """
You are a data modeling assistant.  
The user has provided a query. Your task is to extract:

1. title – A short English identifier to serve as a new column name.  
   • Use only lowercase letters and underscores.  
   • It must start with a letter.  
   • No more than 3 words.  
   • Examples: `root_cause`, `error_family`

2. dimension_desc – A one-sentence description of what data this column should contain, including:  
   • The business meaning this column represents  
   • The expected data type or level of granularity

Requirements  
- You must output a strict Python dict with only two keys: `title` and `dimension_desc`.  
- Do not add any extra fields, comments, or explanations.

Output format example:
{
    "title": "",
    "dimension_desc": ""
}

User query:  
%s
You can also refer to the existing thoughts to adjust the information you need to extract.
%s
"""



class Drill_Down_Agent:
    def __init__(self, llm):
        self.llm = llm

    def understand(self, query, thought):
        prompt = PROMPT_DRILLDOWN_DIMENSION % (query, thought)
        result = self.llm.predict(prompt)
        ret_json = jsonfy_llm_response(result)
        return ret_json

    def run(self, params, node_now):

        dimension_desc = params['desc']
        dimension = params["dimension_name"]
        coarser_dimension = params["related_coarser_dimension"]

        debug_log({
            "type": "log",
            "message": "Drill Down Agent dimension\n```json\n"
                       + json.dumps(dimension, indent=4, ensure_ascii=False) + "\n```"
        })
        tmp_plan = {"operator_name": "sem_map", "parameters":{"columns": [],
                        "map_description": dimension_desc,
                        "keyword": dimension}}
        dimension_params={"title":dimension, "dimension_desc":dimension_desc}
        docs_df = sem_map(self.llm, node_now.docs_df, dimension_params)

        node_now.add_column(dimension, docs_df, dimension_desc, tmp_plan)

        if(coarser_dimension is not None):
            if(coarser_dimension in node_now.col_head.keys()):
                for coarser_granularity in node_now.col_head[coarser_dimension].get_all_nodes():
                    gv = node_now.col_head[coarser_dimension].get_view(coarser_granularity)
                    node_now.col_head[dimension].add_node(coarser_granularity, gv)

                node_now.col_head.pop(coarser_dimension, None)
                return f"New dimension has successfully build: '{dimension}', and dimension '{coarser_dimension}' is merged into it.", docs_df
        return f"New dimension has successfully build: {dimension}", docs_df



