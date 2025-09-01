import streamlit as st
from pyvis.network import Network
import tempfile
import os
st.set_page_config(layout="wide")


def draw_olap_graph(memory):
    net = Network(height="420px", width="100%", directed=True)

    for node_id, node in memory.nodes.items():
        label = f"{node.id}: {node.query}"
        title = f"节点 {node.id}"
        net.add_node(node.id, label=label, title=title)

    for from_id, to_ids in memory.edges.items():
        for to_id in to_ids:
            net.add_edge(from_id, to_id)

    tmp_path = tempfile.mktemp(suffix=".html")
    net.save_graph(tmp_path)
    with open(tmp_path, "r", encoding="utf-8") as f:
        html = f.read()
    os.remove(tmp_path)
    st.components.v1.html(html, height=450, scrolling=False)

def render_pyvis_graph(nodes, edges, height=350):
    g = Network(height=f"{height}px", width="100%", directed=True)
    for node in nodes:
        g.add_node(node, label=node)
    for src, dst in edges:
        g.add_edge(src, dst)

    tmp_path = tempfile.mktemp(suffix=".html")
    g.save_graph(tmp_path)
    with open(tmp_path, "r", encoding="utf-8") as f:
        html = f.read()
    os.remove(tmp_path)
    st.components.v1.html(html, height=height + 30, scrolling=False)

def show_cube_node_details(node):
    filtered_cols = [col for col in node.col_head.keys() if col != "OLAP_ID"]
    tabs = st.tabs(filtered_cols)

    for col, tab in zip(filtered_cols, tabs):
        if(col == "OLAP_ID"):
            continue
        with tab:
            st.markdown(f"##### The View of column `{col}`")

            col_views = node.col_head[col]
            nodes = col_views.get_all_nodes()
            edges = col_views.get_all_edges()

            render_pyvis_graph(nodes, edges, height=300)

            selected_view = st.selectbox(f"Choose a node to check({col})", nodes, key=f"{col}_select")
            gv = col_views.get_view(selected_view)

            st.markdown(f"**View description**: {gv.desc}")
            st.dataframe(gv.df, use_container_width=True)
            if(gv.view is not None):
                st.dataframe(gv.view, use_container_width=True)


st.title("SemaFlex-Cube")

if "memory" not in st.session_state:
    st.session_state.memory = st.session_state["agent"].memory

draw_olap_graph(st.session_state.memory)

st.subheader("Check content of node")

node_options = ["None"] + [
    f"{node.id}: {node.query}" for node in st.session_state.memory.nodes.values()
]
selected_node_label = st.selectbox("Choose a node for detail", node_options)

if selected_node_label != "None":
    selected_id = int(selected_node_label.split(":")[0])
    selected_node = st.session_state.memory.get_node(selected_id)
    show_cube_node_details(selected_node)
else:
    st.info("No node choosed")
