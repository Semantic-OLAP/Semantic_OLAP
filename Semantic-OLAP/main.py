from Agents.OLAP_Agent import OLAP_Agent
import pandas as pd
import json

def flatten_main_records(data):
    def stringify_nested(obj):
        if isinstance(obj, (dict, list)):
            return json.dumps(obj, ensure_ascii=False)
        return obj

    flat_data = []
    for record in data:
        flat_record = {k: stringify_nested(v) for k, v in record.items()}
        flat_data.append(flat_record)

    return pd.DataFrame(flat_data)

def load_json(path="datas/stackoverflow_database_sample_2000.json"):
    with open(path, encoding="utf-8") as f:
        data_json = json.load(f)
    data = flatten_main_records(data_json)
    return data



if (__name__ == "__main__"):
    llm = None
    data = load_json("datas/stackoverflow_database_sample_2000.json")
    olap_agent = OLAP_Agent(llm=llm, data=data)
    query = "Find datas in 2020 year."
    olap_agent.run(query)
