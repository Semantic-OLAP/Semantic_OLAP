"""
This is a lightweight visualization page designed to present data in a clear and intuitive way.
"""





from Agents.OLAP_Agent import OLAP_Agent
import pandas as pd
import streamlit as st
import threading

st.set_page_config(layout="wide")

import socket
import json

def str_to_df_json(s: str) -> pd.DataFrame:
    data = json.loads(s)
    return pd.DataFrame(data)


HOST, PORT = "localhost", 9999
SEND_PORT = 9998

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


def init_agents():
    llm = None
    data = load_json("datas/stackoverflow_database_sample_2000.json")
    olap_agent = OLAP_Agent(llm=llm, data=data)
    return olap_agent


if("historys" not in st.session_state):
    st.session_state["historys"] = []
if("listening" not in st.session_state):
    st.session_state["listening"] = False
if("current_query" not in st.session_state):
    st.session_state["current_query"] = ""
if("current_logs" not in st.session_state):
    st.session_state["current_logs"] = []
if("s" not in st.session_state):
    st.session_state["s"] = None
if("agent" not in st.session_state):
    st.session_state["agent"] = init_agents()

def agent_query(agent, query):
    agent.run(query)

with st.container():
    for record in st.session_state.historys:
        with st.chat_message("user"):
            st.markdown(record["query"])

        with st.expander("log", expanded=False):
            for log in record["logs"]:
                with st.chat_message("assistant"):
                    st.markdown(log)

        with st.chat_message("assistant"):
            st.markdown(record["answer"])

    if st.session_state.listening:
        if(st.session_state["s"] is None):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            s.listen()
            st.session_state["s"] = s


        with st.chat_message("user"):
            st.markdown(st.session_state.current_query)

        with st.expander("logs", expanded=False):
            for log in st.session_state["current_logs"]:
                with st.chat_message("assistant"):
                    st.markdown(log)

        conn, _ = st.session_state["s"].accept()
        data = conn.recv(4096).decode("utf-8")
        for line in data.strip().splitlines():
            try:
                obj = json.loads(line)
                if obj.get("type") == "Answer":
                    st.session_state["s"].close()
                    st.session_state["s"] = None
                    st.session_state["historys"].append({"query": st.session_state["current_query"], "logs": st.session_state["current_logs"], "answer": obj["message"]})
                    st.session_state["current_query"] = ""
                    st.session_state["current_logs"] = []
                    st.session_state["listening"] = False
                else:
                    st.session_state["current_logs"].append(obj["message"])
            except:
                pass
        conn.close()
        st.rerun()

    else:
        prompt = st.chat_input("Say something...")
        if prompt:
            st.session_state.current_query = prompt
            st.session_state.listening = True

            try:
                threading.Thread(target=agent_query, args=(st.session_state["agent"],prompt), daemon=True).start()
            except Exception as e:
                st.session_state.listening = False
            st.rerun()