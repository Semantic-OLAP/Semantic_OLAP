import re
import json
import ast

def jsonfy_llm_response(response, defalt_result=None):
    # 去掉 markdown 代码块标记
    response = re.sub(r"```(?:\w+)?\n(.*?)```", r"\1", response, flags=re.DOTALL).strip()

    # 如果返回的文本前后不是完整 JSON，尝试截取 { ... }
    if response and (response[0].isalnum() or response[-1].isalnum()):
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1 and start < end:
            response = response[start:end + 1]

    # 定义清理函数（只删注释，不动其它）
    def strip_json_comments(text: str) -> str:
        # 去掉 // 注释（行尾）
        text = re.sub(r'//.*?(?=\n)', '', text)
        # 去掉 /* ... */ 注释（多行）
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        return text

    try:
        result = json.loads(response)
    except Exception:
        # 如果失败，检测是否包含注释
        if "//" in response or "/*" in response:
            clean_response = strip_json_comments(response)
            try:
                result = json.loads(clean_response)
            except Exception:
                try:
                    result = ast.literal_eval(clean_response)
                except Exception:
                    print("!" * 50)
                    print(f"ERROR: \n{response}\n can not be parsed into JSON/Python object!")
                    print("!" * 50)
                    result = defalt_result if defalt_result is not None else response
        else:
            # 不含注释，尝试 ast.literal_eval
            try:
                result = ast.literal_eval(response)
            except Exception:
                print("!" * 50)
                print(f"ERROR: \n{response}\n can not be parsed into JSON/Python object!")
                print("!" * 50)
                result = defalt_result if defalt_result is not None else response

    return result
