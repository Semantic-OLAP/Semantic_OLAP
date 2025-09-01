from Agents.Dice_Agent import Dice_Agent
from Agents.Slice_Agent import Slice_Agent
from Agents.Drill_Down_Agent import Drill_Down_Agent
from Agents.Roll_Up_Agent import Roll_Up_Agent
from Agents.Exection_Agent import Execution_Agent
from Agents.Components.OLAP_Memory import OLAP_Memory
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict, Union
import os
import pandas as pd
from Utils.send_logs import debug_log
import re
from Utils.jsonfy_result import jsonfy_llm_response
from Agents.Components.Operaters import sem_topk, num_topk



class State(TypedDict):
    query: str
    bot_view: str


PLAN_OLAP_AGENT_PROMPT_SLICE_DICE = """
You are a query planner responsible for breaking down a user's natural language query intent into a set of atomic filtering steps, and assigning them to one of two agents in the system:

1. dice agent: Performs filtering on a clearly identified single field.  
   Appropriate when: the query explicitly mentions a specific field, or the filter value clearly maps to one known field.

2. slice agent: Performs filtering across all fields.  
   Appropriate when: the filter condition cannot be mapped to a specific field, or it may involve multiple fields and requires cross-field or full-text search.

[Input Information]  
- List of structured fields (these are all directly available to the dice agent):  
%s  
- Historical Query (the user’s previous query):  
%s  
- Current Query:  
%s  

[Task Instructions]  
- The current query builds upon the historical query and represents a refinement. Start by identifying what new filtering conditions have been added compared to the historical query.
    Step 1: Compare historical query and current query meaning-by-meaning, and extract only the new filter conditions that are NOT already covered in the historical query.
    Step 2: Use only these new conditions in the planning below.
- Analyze the semantic intent of the current query and determine which filtering conditions can be handled by the dice agent (targeting a known single field), and which require the slice agent (cross-field or ambiguous field).
- Each operation must include only one filtering condition. Do not merge multiple conditions into a single operation.
- Break down the query intent into a sequence of atomic single-condition filtering steps, and define their logical combination.

[Output Requirements]  
The output must be strictly valid JSON, with no extra text. It should include:
- `filter_condition`: The filtering conditions added in the current query (compared to the historical query)
- `reasoning`: A brief explanation of how the operations and logical structure were determined from the added conditions
- `operations`: A list of filtering steps, each including:  
  - `id`: A unique integer ID starting from 1  
  - `agent`: Either `"dice"` or `"slice"`  
  - `instruction`: A brief human-readable description of the filtering step (must clearly indicate this one condition) 
  - `field`: If the agent is `"dice"`, this should be the corresponding field name from the structured field list. If the agent is `"slice"`, set this to `null`.  
  - `action`: A natural language instruction describing what kind of content the agent should keep. In other words, specify what should be retained after filtering.

- `logic`: A nested array representing the logical structure of the operations  
  - The first element is a logical operator `"AND"` or `"OR"`  
  - The remaining elements are either operation IDs or nested sub-expressions (arrays)

Note: In each `action`, focus on clearly stating what needs to be filtered according to the user query. Do not specify how the agent should perform the task — simply restate the filtering condition in full natural language.

[Example Output Format]  
{   
  "historical_conditions": "The filtering conditions already applied from the historical query.",
  "current_conditions": "All filtering conditions implied by the current query, including any carried over from the historical query.",
  "filter_condition": "The additional filter(s) found by comparing current_conditions to historical_conditions — these are the new constraints to apply on top of the existing filters.",
  "reasoning": "Explain how the identified filter_condition (difference between current_conditions and historical_conditions) is used to plan the operations. Ensure that historical_conditions are not repeated in the operations.",
  "operations": [
    {
      "id": 1,
      "agent": "dice",
      "instruction": "Describe step 1",
      "field": "Choosed field name if dice, or 'null' if slice",
      "action": "Describe the filtering condition only, without mentioning field, agent, or any other meta information."
    },
    ...
  ],
  "logic": ["OR", ["AND", 1, 2], 3]
}
"""


def make_stepwise_ReAct_prompt(query: str, history: str = "") -> str:
    return """You are assisting in building a knowledge organization system that operates on a structured data object.
[Data Structure]
This data is organized across multiple dimensions and granularities:
A dimension represents a semantic axis along which data can be viewed or grouped.
Each dimension contains one or more granularities, which define the level of abstraction or detail for that dimension — for example, a detailed textual explanation may represent a finer granularity, while a broader category label represents a coarser one.
Each granularity specifies how every record expresses that dimension at its level — in other words, it defines the concrete values that each data record will have for that dimension at that abstraction level.
Each dimension always starts with a granularity of the same name, which represents its initial level of detail.

[Your Task]
Your task is to gradually plan or refine the structure so that it can support the user’s query intent. Ignore any filtering or sorting (top-k) requirements, as they are handled elsewhere.
Using the available actions, construct the semantic structure needed to answer the query.
You should follow a ReAct-style process: Thought → Action → Observation, and only output the next step at each round.
If the current structure is already sufficient for the query, return "action": null to end the reasoning loop.
Default principle: Make the minimal structural change necessary to answer the query. 
After every Observation, FIRST check sufficiency: 
- If the current views already let you directly answer the query, return "action": null. 
- Do NOT add new granularities or perform aggregation unless strictly required by the user’s intent.



[Available Actions]
- get_dimension
This is a lookup action that returns all existing dimensions in the data structure.
"params": {} # No parameters required

- get_granularity
This is a lookup action that, given a specific dimension, returns all granularities available under that dimension.
"params": { "dimension": "<the name of the dimension>" }


- drill_down
Drill-down is used to create a new dimension that either:
1. Is orthogonal to all existing dimensions — meaning it introduces a completely new semantic axis not covered by the current structure.
2. Or, for all existing dimensions, the required dimension’s granularity is finer than theirs, and cannot be obtained by grouping from any existing dimension’s values to a higher-level abstraction. In other words, it cannot be derived by re-expressing existing dimensions at a coarser or alternative abstraction level, but must be directly extracted from the raw data.
Note:
Treat the new dimension as orthogonal by default.
Only consider it related to an existing dimension if there is a clear coarse-to-fine granularity relationship, and the existing dimension’s values can be obtained by grouping the new dimension’s values into those categories.
Execution rules:
- Case 1 (orthogonal): 
    a. Create a completely new dimension with the given name and description.
    b. This dimension is independent of all existing dimensions, so no changes are made to other dimensions or their granularities.
- Case 2 (finer granularity of an existing dimension): 
    a. Create a new dimension with the given name and description.
    b. Move the related existing dimension into this new dimension as one of its granularities.
    c. Remove the original granularity from the related dimension, ensuring the structure remains consistent.
"params":
{
  "desc": "<A natural language description of the new dimension and what it represents>",
  "dimension_name": "<The name of the new dimension to be created>",
  "related_coarser_dimension": "<Name of an existing related but coarser dimension, or 'None' if no such dimension exists (orthogonal case)>"
}

- roll_up
Roll-up is used to transform an existing dimension into a coarser granularity by grouping its current values into higher-level categories.  
Each group should have a representative label (target granularity value) that reflects the common characteristics of the grouped items.
You only need to provide the target_granularity name. The system will automatically create the new granularity, generate its values, and populate them. You do not need to manually create it through drill_down or any other action — the system will handle this automatically.
Optionally, after grouping, you can perform additional analysis:
1. Cross-dimension analysis — examine how the grouped results relate to one or more other dimensions.
2. Self-analysis — analyze the grouped dimension itself (only aggregate-level statistics such as count or numeric reductions are possible, because all values in each group are identical after grouping).
Execution rules:
If no grouping is needed (i.e., the current granularity is already appropriate and only analysis is required), set target_granularity to None.
1. The system will group the records of the specified dimension into the categories defined by the target granularity, producing one aggregated group for each target value. And then create a new granularity for it named <target_granularity>.
2. Within each group, the system will assign the target granularity label that represents the shared characteristics of its members.
3. (Optional) For each analysis target in the list:
    - If it refers to another dimension: the system will perform the specified reduction between the grouped dimension and that other dimension.
    - If it is "self": the system will count the records in each group, or, if the grouped values are numeric, perform a numeric reduction (sum, average, min, max, etc.).
"params":
{
  "dimension": "<The dimension whose values will be grouped into a coarser granularity>",
  "target_granularity": "<The name of the new coarser granularity to roll up to, or None if no grouping is needed>",
  "analyze_dimension": [
    {
      "thought": "<Explain why this analysis is needed or what insight is expected>",
      "dimension": "<Name of another dimension to analyze, or 'self' to analyze the grouped dimension itself>",
      "reduce_target": "Describe what kind of analysis to perform. For 'self', this can only be counting the records in each group, or applying a numeric reduction (such as sum, average, minimum, or maximum) if the grouped values are numeric. For other dimensions, this can be a semantic analysis (e.g., classifying, extracting topics) or a numerical aggregation, depending on what is explicitly stated in the query. If the query does not specify, set this to None."
    },
    ...
  ]
}

[Output Format]

If a next step is needed:
{
  "thought": "A brief explain of your choice",
  "action": {
    "type": "<one of: drill_down, roll_up, get_dimension, get_granularity>",
    "params": { ... }
  }
}

If the structure is already sufficient:
{
  "thought": "No further refinement needed. The current views are sufficient.",
  "action": null
}


Input:
- User query:
%s

- History:
%s
""" % (query, (history or ""))


UNDERSTAND_TOPK_PROMPT = """
You are a query analyzer responsible for determining whether a user's natural language query contains a top-k intent, and if so, standardizing the extracted information.

[Task Instructions]
1. Determine whether the query expresses top-k intent.
- Top-k intent is typically indicated by phrases like: top 5, first, most, highest, etc.
- Ignore general sorting words if they do not explicitly indicate a top-N selection.

2. If top-k intent is found:
- topk_type: Choose "num" when the ranking is based on a measurable or computed quantity from the structured field candidates.  
  Choose "sem" when the ranking is based on qualitative, descriptive, or subjective characteristics.
- sort_field: The field to rank by. Must be one of the structured field candidates above.
- sort_order: 'desc' if the query implies largest/highest/most; 'asc' if it implies smallest/lowest/least.
- top_k: The numeric value of N (e.g., 5, 10).
- sort_basis: A natural language description explaining the basis of sorting as implied by the query.

3. If no top-k intent is found, return a simple flag.

[Output Requirements]
Return strictly valid JSON with no extra text.

If top-k intent is found:
{
  "topk_type": "<choose 'num' for numeric top-k, or 'sem' for semantic top-k>",
  "sort_field": "<name of the field from structured list>",
  "sort_order": "desc",
  "top_k": <number>,
  "sort_basis": "<natural language description of the sorting basis>"
}

If not found:
{
  "topk_type": "not applicable"
}


[Input Information]
- Structured field candidates that the user may want to rank by:  
%s  

- User Query:  
%s  
"""


class OLAP_Agent:
    def __init__(self, llm: Any, data: pd.DataFrame, memory: OLAP_Memory = None):
        self.llm = llm
        self.slice_agent = None
        self.data = data

        if (memory is not None):
            self.memory = memory
        else:
            self.memory = OLAP_Memory(llm)
            self.memory.init(self.data)

        self.dice_agent = Dice_Agent(llm)
        self.slice_agent = Slice_Agent(llm)
        self.rollup_agent = Roll_Up_Agent(llm)
        self.drilldown_agent = Drill_Down_Agent(llm)
        self.Execution_Agent = Execution_Agent(llm)

        self.agent_map = {"dice": self.dice_agent,
                          "slice": self.slice_agent,
                          "roll_up": self.rollup_agent,
                          "drill_down": self.drilldown_agent}

    def decompose_query_intent(self, query):
        prompt = """
You are a query decomposition assistant. Your task is to break down a user's natural language query into the following three semantically distinct sub-queries:

1. filter_query: Describes the subset of data the user wants to select. This typically includes constraints on time, entities, status, numerical ranges, etc. The essence is: “which data points are of interest.” This part narrows down the dataset by reducing the number of rows but does not change the structure of the data.

2. analysis_query: Describes how the user wants to organize, aggregate, or transform the selected data. This includes exploring a new dimension or group some dimensions into coarser granularity. The essence is: “how to process the selected data structurally.” This part may change the structure or granularity of the data, but does not affect the filtered scope.

Decomposition rules:
- If the original query does not contain a certain type of intent, return an empty string "" for that part.
- Do not provide any explanation — only return the output in the following format:

{
  "filter_query": "...",
  "analysis_query": "..."
}

Query: %s
""" % query

        response = self.llm.predict(prompt)

        try:
            parsed = jsonfy_llm_response(response)
            return parsed.get("filter_query", "").strip(), parsed.get("analysis_query", "").strip()
        except Exception:
            return "", "", ""

    def plan_generate_filter(self, history_query, now_query, node_now):
        col_str = str(node_now.col_head.keys())
        prompt = PLAN_OLAP_AGENT_PROMPT_SLICE_DICE % (col_str, history_query, now_query)
        result_str = self.llm.predict(prompt)
        result = jsonfy_llm_response(result_str)
        return result

    def execute_plan_filter(self, plan, node_now):
        new_operations = []

        for op in plan["operations"]:
            agent = op['agent']
            action = op['action']
            field = op['field']
            agent_now = self.agent_map[agent]

            query = {'action': action, 'field': field}
            sub_plan = agent_now.run(query, node_now)

            op_with_plan = {
                "id": op["id"],
                "plan": sub_plan
            }

            new_operations.append(op_with_plan)

        new_plan = {
            "operations": new_operations,
            "logic": plan["logic"]
        }

        return new_plan

    def run_roll_up_and_drill_down(self, query_view, node_now):
        END_TAG = 0
        history = ""
        df_now = node_now.docs_df
        while END_TAG < 10:
            END_TAG += 1

            prompt = make_stepwise_ReAct_prompt(query_view, history)
            result = self.llm.predict(prompt)
            ret_json = jsonfy_llm_response(result)

            if not ret_json or ret_json.get("action") is None:
                break

            action = ret_json["action"]
            action_type = action["type"]
            params = action.get("params", {})
            params["thought"] = ret_json["thought"]
            if action_type == "drill_down":
                observation, df_now = self.drilldown_agent.run(params, node_now)
            elif action_type == "roll_up":
                observation, df_now = self.rollup_agent.run(params, node_now)
            elif action_type == "get_dimension":
                observation = node_now.get_dimension(params)
            elif action_type == "get_granularity":
                observation = node_now.get_granularity(params)
            else:
                break

            history_now = {
                "thought": ret_json['thought'],
                "action": json.dumps(action, ensure_ascii=False),
                "observation": observation
            }

            history += (
                f"thought: {ret_json['thought']}\n"
                f"action: {json.dumps(action, ensure_ascii=False)}\n"
                f"observation: {observation}\n"
            )

            debug_log({
                "type": "log",
                "message": f"OLAP Agent: \n\n ```json\n{json.dumps(history_now, indent=4, ensure_ascii=False)}\n```"
            })

        return df_now


    def run(self, query):

        new_query = query

        text = f"OLAP Agent: query:\n\n `{new_query}`"
        obj = {"type": "log", "message": text}
        debug_log(obj)

        filter_query, analysis_query = self.decompose_query_intent(new_query)
        querys = {'filter_query': filter_query, 'analysis_query': analysis_query}
        print(querys)

        text = f"OLAP Agent: query\n\n```json\n{json.dumps(querys, ensure_ascii=False, indent=4)}\n```"
        obj = {"type": "log", "message": text}
        debug_log(obj)

        nodes, state = self.memory.get_current_node(filter_query)

        node_now_id = nodes[0]
        node_now = self.memory.get_node(node_now_id)
        text = f"OLAP Agent: locate node:\n\n `{node_now.query}`"
        obj = {"type": "log", "message": text}
        debug_log(obj)

        show_table = None

        if (state != "Equal"):
            if (nodes == []):
                nodes = [0]
            node_id = nodes[0]
            node = self.memory.get_node(node_id)
            history_query = node.query

            plan_filter = self.plan_generate_filter(history_query, filter_query, node_now)

            operations_filter = self.execute_plan_filter(plan_filter, node_now)

            result_df, tmp_plan = self.Execution_Agent.run_filter(operations_filter, node_now)
            id = self.memory.add_node(filter_query, result_df)
            for node in nodes:
                self.memory.add_edge(node, id)
            if (node_now.plan is not None):
                parent_plan = node_now.plan
            else:
                parent_plan = []
            node_now = self.memory.get_node(id)
            node_now.plan = parent_plan + tmp_plan
            show_table = node_now.docs_df

        if (analysis_query != ""):
            show_table = self.run_roll_up_and_drill_down(analysis_query, node_now)

            descriptions = []
            for col in show_table.columns:
                if (col == "OLAP_ID"):
                    continue
                sample_vals = show_table[col].dropna().head(2).tolist()
                total_chars = sum(len(s) for s in sample_vals if isinstance(s, str))

                if total_chars > 100:
                    descriptions.append(f"Field: {col} | Samples: content too long to display")
                else:
                    descriptions.append(f"Field: {col} | Samples: [{', '.join(map(str, sample_vals))}]")

            table_str = "\n".join(descriptions)
            print(table_str)
            prompt_topk = UNDERSTAND_TOPK_PROMPT % (table_str, analysis_query)
            result = self.llm.predict(prompt_topk)
            print(result)
            topk_params = jsonfy_llm_response(result)
            if (topk_params["topk_type"] in ["num", "sem"]):
                if topk_params["topk_type"] == "sem":
                    show_table = sem_topk(
                        llm=self.llm,
                        df=show_table,
                        columns=[topk_params["sort_field"]],
                        query=topk_params["sort_basis"] + f"\n Order: {topk_params['sort_order']}",
                        k=topk_params["top_k"]
                    )

                else:
                    if (is_column_numeric(show_table, topk_params["sort_field"])):
                        show_table = num_topk(
                            df=show_table,
                            column=topk_params["sort_field"],
                            k=topk_params["top_k"],
                            order=topk_params["sort_order"]
                        )
                    else:
                        show_table = sem_topk(
                            llm=self.llm,
                            df=show_table,
                            columns=[topk_params["sort_field"]],
                            query=topk_params["sort_basis"] + f"\n Order: {topk_params['sort_order']}",
                            k=topk_params["top_k"]
                        )

        text = "END~~"
        obj = {"type": "Answer", "message": text}
        debug_log(obj)


def is_column_numeric(df, col_name):
    series = df[col_name].dropna()
    numeric_series = pd.to_numeric(series, errors='coerce')
    return not numeric_series.isna().any()


import json


def df_to_str_json(df: pd.DataFrame) -> str:
    return df.to_json(orient='records', force_ascii=False)
