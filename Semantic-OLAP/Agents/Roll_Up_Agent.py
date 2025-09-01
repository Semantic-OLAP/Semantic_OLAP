from sentence_transformers import SentenceTransformer

import pandas as pd
import json
from Agents.Components.OLAP_Memory import Granularity_View
from Utils.send_logs import debug_log
from Agents.Components.Operaters import sem_group, group_by, sem_reduce, count, num_reduce
from Utils.jsonfy_result import jsonfy_llm_response


def understand_self_reduce(llm, description):
    prompt = """
You are a data reduction task parser.
Given a short task description, determine:
1. Task type: "count" or "num_reduce".
2. If type is "count", return only:
   {
     "type": "count"
   }
3. If type is "num_reduce", return:
   {
     "type": "num_reduce",
     "params": {
       "agg": "sum | avg | max | min"
     }
   }
Rules:
- "count" means counting rows or occurrences.
- "num_reduce" means numeric aggregation such as sum, average, max, min.
- Output valid JSON only.

Task description:
%s
""" % description

    result = llm.predict(prompt)
    return jsonfy_llm_response(result)

def reduce_groups_to_dataframe(llm, grouped_docs: dict, group_key_col: str, analyze_plan: list):
    rows = []

    col_ops = {}
    for step in analyze_plan:
        op = step["operator_name"]
        params = step.get("parameters", {})
        if op in ("count", "num_reduce") and params.get("group_by") == group_key_col:
            col_ops[group_key_col] = step
        elif op == "sem_reduce":
            for col in params.get("columns", []):
                col_ops[col] = step

    for group_key, group_df in grouped_docs.items():
        row = {group_key_col: group_key}

        for col in group_df.columns:
            if col == group_key_col and col in col_ops:
                step = col_ops[col]
                if step["operator_name"] == "count":
                    row[f"count_of_{group_key_col}"] = count(group_df[group_key_col])
                elif step["operator_name"] == "num_reduce":
                    agg = step["parameters"].get("agg")
                    row[f"{agg}_of_{group_key_col}"] = num_reduce(group_df[group_key_col], agg)
                continue

            if col == "OLAP_ID":
                row[col] = group_df[col].tolist()
                continue

            if col in col_ops and col_ops[col]["operator_name"] == "sem_reduce":
                step = col_ops[col]
                row[col] = sem_reduce(llm, col, group_df[col], step["parameters"])

        rows.append(row)

    return pd.DataFrame(rows)





def query_dimension_exist(llm, dimension_node, query_granularity, thought):
    granularities = list(dimension_node._nodes.keys())
    if (query_granularity in granularities):
        return query_granularity

    prompt = """
Please select the granularity from the list that is most closely aligned with the query granularity in meaning.
A match should only be made if they refer to the same conceptual level of detail. 
If there is no such match, return "None".

Return your answer as a JSON object with the following keys:
- "chosen": the chosen granularity exactly as it appears in the list, or "None" if none match in both meaning and level of detail.
- "thought": a short explanation of your reasoning.

The output should follow the following format:
{
  "thought": "<Your thought>",
  "chosen": "<Your choice>"
}

You can also refer to the existing thoughts to adjust your determination:
%s

[Available granularities]
%s

[Query granularity]
%s

Only return a valid JSON object. Do not return any other text.
""" % (thought, granularities, query_granularity)

    result_str = llm.predict(prompt).strip()

    try:
        result_json = jsonfy_llm_response(result_str)
    except json.JSONDecodeError:
        return None
    chosen = result_json.get("chosen")

    if chosen in granularities:
        return chosen
    elif chosen == "None":
        return None
    else:
        return None


class Roll_Up_Agent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, params, node_now):
        print("=====roll up=====")
        print(params)
        dimension = params["dimension"]
        target_granularity = params["target_granularity"]
        analyze_dimension = params["analyze_dimension"]

        now_dimension = node_now.col_head[dimension]
        exist_granularity = None
        if (target_granularity is not None and target_granularity!="None"):
            exist_granularity = query_dimension_exist(self.llm, now_dimension, target_granularity, params["thought"])
            print(exist_granularity)
            if (exist_granularity is not None):
                actual_used_granularity = exist_granularity
                new_docs_df = node_now.col_head[dimension].get_view(exist_granularity).df
            else:
                actual_used_granularity = target_granularity
                now_granularity = node_now.col_head[dimension].root
                df_now = node_now.col_head[dimension].get_view(now_granularity).df
                new_docs_df = sem_group(llm=self.llm, df=df_now, target=target_granularity, text_col=now_granularity)

                node_now.col_head[dimension].add_node(name=target_granularity,
                                                      gv=Granularity_View(df=new_docs_df, desc=target_granularity))

                node_now.col_head[dimension].get_view(target_granularity).plan = [{
                    "operator_name": "sem_group",
                    "parameters": {
                        "columns": [
                            now_granularity
                        ],
                        "group_description": f"group into granularity {target_granularity}",
                        "keyword": target_granularity
                    }
                }]
        else:
            actual_used_granularity = dimension
            new_docs_df = node_now.col_head[dimension].get_view(actual_used_granularity).df
        if (analyze_dimension == []):
            columns_to_extract = [d for d in node_now.col_head.keys() if d != dimension]
            df_extracted = node_now.docs_df[columns_to_extract]
            merged_df = pd.merge(df_extracted, new_docs_df, on="OLAP_ID", how="right")
            if actual_used_granularity == dimension:
                return f"No roll-up performed. Using the existing granularity '{dimension}' directly for analysis.", merged_df

            elif exist_granularity is not None and actual_used_granularity == exist_granularity:
                return f"The target granularity '{target_granularity}' already exists in '{dimension}' as '{exist_granularity}', using it directly.", merged_df

            else:
                return f"Created new granularity '{target_granularity}' for '{dimension}' and grouped data accordingly.", merged_df

        analyze_plan = []
        columns_analysis = {}
        self_tag = False
        for analyze in analyze_dimension:
            if (analyze["dimension"] == "self"):
                self_tag = True
                self_plan = understand_self_reduce(self.llm, analyze["reduce_target"])
                if (self_plan["type"] == "count"):
                    analyze_plan.append({
                        "operator_name": "count",
                        "parameters": {
                            "group_by": actual_used_granularity,
                            "columns": []
                        }
                    })
                else:
                    analyze_plan.append({
                        "operator_name": "num_reduce",
                        "parameters": {
                            "group_by": actual_used_granularity,
                            "columns": [],
                            "agg": self_plan["params"]["agg"]
                        }
                    })
            else:
                if(analyze["dimension"] in node_now.col_head.keys() and analyze["dimension"] != dimension):
                    columns_analysis[analyze["dimension"]]=analyze["reduce_target"]

        if(len(list(columns_analysis.keys()))>0):
                analyze_plan.append({
                    "operator_name": "sem_reduce",
                    "parameters": {
                        "columns": list(columns_analysis.keys()),
                        "group_by": actual_used_granularity
                    }
                })

        columns_to_extract = ["OLAP_ID"] + list(columns_analysis.keys())
        df_extracted = node_now.docs_df[columns_to_extract]
        common_cols = df_extracted.columns.intersection(new_docs_df.columns).tolist()
        common_cols = [c for c in common_cols if c != "OLAP_ID"]
        df_extracted = df_extracted.drop(columns=common_cols)
        merged_df = pd.merge(df_extracted, new_docs_df, on="OLAP_ID", how="right")

        group_docs = group_by(merged_df, actual_used_granularity)

        summary_df = reduce_groups_to_dataframe(
            llm=self.llm,
            grouped_docs=group_docs,
            group_key_col=actual_used_granularity,
            analyze_plan=analyze_plan
        )
        node_now.col_head[dimension].get_view(actual_used_granularity).set_view(summary_df)
        node_now.col_head[dimension].get_view(actual_used_granularity).plan += analyze_plan


        analyzed_columns = list(columns_analysis.keys())
        if(self_tag):
            analyzed_columns.append("self")

        if actual_used_granularity == dimension:
            return f"No roll-up performed. Using the existing granularity '{dimension}' directly for analysis. And Then '{analyzed_columns}' has been analyzed according to it.", summary_df

        elif exist_granularity is not None and actual_used_granularity == exist_granularity:
            return f"The target granularity '{target_granularity}' already exists in '{dimension}' as '{exist_granularity}', using it directly. And Then '{analyzed_columns}' has been analyzed according to it.", summary_df

        else:
            return f"Created new granularity '{target_granularity}' for '{dimension}' and grouped data accordingly. And Then '{analyzed_columns}' has been analyzed according to it.", summary_df
