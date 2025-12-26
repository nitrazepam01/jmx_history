import streamlit as st
import pandas as pd
import re
import time
from supabase import create_client, Client
from openai import OpenAI

# -----------------------------------------------------------------------------
# 1. 配置与样式优化 (Configuration & Styling)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="JMX近代史刷题助手", layout="centered")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stButton button {
        width: 100%;
        border-radius: 12px;
        height: 50px;
        font-weight: bold;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .question-text {
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 20px;
        line-height: 1.6;
        color: #333;
    }
    
    .stAlert {
        border-radius: 10px;
    }
    
    /* 错题本专用样式 */
    .mistake-badge {
        background-color: #ff4b4b;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 14px;
        font-weight: bold;
        margin-bottom: 10px;
        display: inline-block;
    }
    
    /* 统计区域样式 */
    .stat-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. 数据加载与解析 (Data Loading & Parsing)
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_parse_data(file_path):
    try:
        df = pd.read_csv(file_path, header=None)
    except Exception as e:
        st.error(f"读取 CSV 文件失败: {e}")
        return []

    parsed_data = []
    for idx, row in df.iterrows():
        raw_text = str(row[0])
        correct_ans = str(row[1]).strip().upper()
        
        parts = re.split(r'<br>\s*<br>', raw_text, maxsplit=1)
        if len(parts) < 2:
            question_text = raw_text
            options_block = ""
        else:
            question_text = parts[0].strip()
            options_block = parts[1].strip()

        raw_options = re.split(r'<br\s*/?>', options_block)
        options_dict = {}
        opt_pattern = re.compile(r'^\s*([A-D])\.\s*(.*)', re.DOTALL)
        
        for opt_str in raw_options:
            match = opt_pattern.match(opt_str.strip())
            if match:
                key = match.group(1)
                val = match.group(2)
                options_dict[key] = val

        parsed_data.append({
            "index": idx,
            "question": question_text,
            "options": options_dict,
            "answer": correct_ans
        })
    return parsed_data

questions_data = load_and_parse_data("courseware.csv")

# -----------------------------------------------------------------------------
# 3. 后端服务 (Supabase & AI)
# -----------------------------------------------------------------------------
USER_ID = "user_01"

def init_supabase():
    # 增加容错检查，防止未配置 Secrets 报错
    if "SUPABASE_URL" not in st.secrets:
        st.error("请在 Streamlit Cloud Settings -> Secrets 中配置 SUPABASE_URL")
        st.stop()
        
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()

def get_user_history():
    try:
        response = supabase.table("user_attempt_history")\
            .select("question_index, is_correct, timestamp")\
            .eq("user_id", USER_ID)\
            .execute()
        
        data = response.data
        if not data:
            return {}

        df_hist = pd.DataFrame(data)
        df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'])
        latest_attempts = df_hist.sort_values('timestamp').drop_duplicates('question_index', keep='last')
        status_map = dict(zip(latest_attempts['question_index'], latest_attempts['is_correct']))
        return status_map
    except Exception as e:
        st.error(f"数据库连接错误: {e}")
        return {}

def log_attempt(q_index, selected_opt, is_correct):
    try:
        supabase.table("user_attempt_history").insert({
            "user_id": USER_ID,
            "question_index": q_index,
            "selected_option": selected_opt,
            "is_correct": is_correct
        }).execute()
    except Exception as e:
        st.error(f"保存进度失败: {e}")

def get_ai_explanation(question, user_choice, correct_choice):
    try:
        api_key = st.secrets.get("DEEPSEEK_API_KEY") or st.secrets["OPENAI_API_KEY"]
        base_url = "https://api.deepseek.com" if "DEEPSEEK_API_KEY" in st.secrets else None
        model_name = "deepseek-chat" if "DEEPSEEK_API_KEY" in st.secrets else "gpt-3.5-turbo"

        client = OpenAI(api_key=api_key, base_url=base_url)
        
        prompt = f"""
        用户选错了。题目: "{question}"
        用户选: "{user_choice}"
        正确答案: "{correct_choice}"
        请解释：1. 为什么选错了(常见误区)。2. 为什么正确答案是对的。语气要亲切鼓励。
               2.告诉用户这里怎么记忆最简单那
               3.用“你好，姜同学”开头 给她一句简短的鼓励
        """
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI 解析暂时不可用: {e}"

# -----------------------------------------------------------------------------
# 4. 状态管理
# -----------------------------------------------------------------------------
if 'current_q_index' not in st.session_state:
    st.session_state.current_q_index = 0
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'grid' # 'grid', 'quiz', 'review_mistakes'
if 'explanation' not in st.session_state:
    st.session_state.explanation = None
if 'mistake_pointer' not in st.session_state:
    st.session_state.mistake_pointer = 0

# -----------------------------------------------------------------------------
# 5. 视图 1: 题库概览 (Dashboard)
# -----------------------------------------------------------------------------
if st.session_state.view_mode == 'grid':
    st.title("🧩 题库概览")
    
    status_map = get_user_history()
    total_q = len(questions_data)
    completed = len(status_map)
    correct_count = sum(1 for v in status_map.values() if v)
    
    # === 计算逻辑：寻找“最新进度” ===
    # 找到第一个没有在 status_map 中出现的 ID
    next_todo_index = 0
    for i in range(total_q):
        if i not in status_map:
            next_todo_index = i
            break
    else:
        # 如果循环走完没 break，说明全做完了
        next_todo_index = -1 

    # === 计算逻辑：错题 ===
    wrong_indices = [idx for idx, is_right in status_map.items() if not is_right]
    wrong_indices.sort()
    wrong_count = len(wrong_indices)

    # 顶部统计
    col1, col2, col3 = st.columns(3)
    col1.metric("已完成", f"{completed}/{total_q}")
    col2.metric("正确率", f"{int(correct_count/completed*100)}%" if completed > 0 else "0%")
    col3.metric("错题本", f"{wrong_count}", delta_color="inverse")
    
    st.markdown("---")

    # === 操作区 (Action Bar) ===
    # 按钮 1: 继续做题 (仅当没做完时显示)
    if next_todo_index != -1:
        # 这里使用 Primary 样式，使其最显眼
        if st.button(f"🚀 继续刷题 (从第 {next_todo_index + 1} 题开始)", type="primary", use_container_width=True):
            st.session_state.current_q_index = next_todo_index
            st.session_state.view_mode = 'quiz'
            st.session_state.explanation = None
            st.rerun()
    else:
        st.success("🎉 哇！你已经刷完所有题目啦！太强了！")

    # 按钮 2: 复习错题 (仅当有错题时显示)
    if wrong_count > 0:
        # 增加一点间距
        st.write("") 
        if st.button(f"📖 专项复习错题 ({wrong_count}题)", use_container_width=True):
            st.session_state.view_mode = 'review_mistakes'
            st.session_state.mistake_pointer = 0
            st.session_state.explanation = None
            st.rerun()

    st.markdown("### 所有题目")
    
    # 题目网格
    cols_per_row = 5
    rows = [questions_data[i:i + cols_per_row] for i in range(0, total_q, cols_per_row)]
    
    for row in rows:
        cols = st.columns(cols_per_row)
        for idx, q_item in enumerate(row):
            q_idx = q_item['index']
            
            btn_type = "secondary"
            btn_label = f"{q_idx + 1}"
            
            if q_idx in status_map:
                if status_map[q_idx]:
                    btn_label = f"✅ {q_idx + 1}"
                else:
                    btn_label = f"❌ {q_idx + 1}"
                    # 网格里也把错题高亮一下
                    btn_type = "primary" 
            
            with cols[idx]:
                if st.button(btn_label, key=f"grid_btn_{q_idx}", type=btn_type, use_container_width=True):
                    st.session_state.current_q_index = q_idx
                    st.session_state.view_mode = 'quiz'
                    st.session_state.explanation = None
                    st.rerun()

# -----------------------------------------------------------------------------
# 6. 通用答题组件
# -----------------------------------------------------------------------------
def render_quiz_ui(q_idx, is_review_mode=False, total_wrong_count=0, current_wrong_pos=0):
    q_data = questions_data[q_idx]
    
    if st.button("⬅️ 返回主页"):
        st.session_state.view_mode = 'grid'
        st.session_state.explanation = None
        st.rerun()

    if is_review_mode:
        st.markdown(f"<div class='mistake-badge'>🔥 错题突击: 第 {current_wrong_pos + 1} / {total_wrong_count} 个</div>", unsafe_allow_html=True)
    
    st.markdown(f"<div class='question-text'>{q_idx + 1}. {q_data['question']}</div>", unsafe_allow_html=True)
    
    options = q_data['options']
    if not options:
        st.warning("选项解析失败")
        return

    option_labels = [f"{k}. {v}" for k, v in options.items()]
    
    radio_key = f"radio_{q_idx}_review" if is_review_mode else f"radio_{q_idx}"
    
    selected_label = st.radio(
        "请选择答案:",
        option_labels,
        index=None,
        key=radio_key
    )
    
    submit_col, next_col = st.columns([1, 1])
    
    if st.session_state.explanation is None:
        with submit_col:
            if st.button("提交答案", type="primary", use_container_width=True):
                if selected_label:
                    user_choice_key = selected_label.split(".")[0]
                    correct_key = q_data['answer']
                    is_correct = (user_choice_key == correct_key)
                    
                    log_attempt(q_idx, user_choice_key, is_correct)
                    
                    if is_correct:
                        st.balloons()
                        msg = "✅ 答对了！该题已消灭！" if is_review_mode else "✅ 答对了！"
                        st.success(msg)
                        time.sleep(1.0)
                        
                        if is_review_mode:
                            st.rerun() 
                        else:
                            # 普通模式自动下一题
                            if st.session_state.current_q_index < len(questions_data) - 1:
                                st.session_state.current_q_index += 1
                                st.rerun()
                            else:
                                st.success("全题库已刷完！")
                                time.sleep(2)
                                st.session_state.view_mode = 'grid'
                                st.rerun()
                    else:
                        st.error(f"我草、用户做错了。正确答案是 {correct_key}。")
                        with st.spinner("🤖Deepseek老师正在分析..."):
                            expl = get_ai_explanation(
                                q_data['question'], 
                                options.get(user_choice_key, "未知"), 
                                options.get(correct_key, "未知")
                            )
                            st.session_state.explanation = expl
                            st.rerun()
                else:
                    st.warning("请选择一个选项")

    if st.session_state.explanation:
        st.info(f"**🤖 AI 解析:**\n\n{st.session_state.explanation}")
        with next_col:
            btn_text = "下一道错题 ➡️" if is_review_mode else "下一题 ➡️"
            if st.button(btn_text, type="primary", use_container_width=True):
                st.session_state.explanation = None
                
                if is_review_mode:
                    st.session_state.mistake_pointer += 1
                    st.rerun()
                else:
                    if st.session_state.current_q_index < len(questions_data) - 1:
                        st.session_state.current_q_index += 1
                        st.rerun()
                    else:
                        st.session_state.view_mode = 'grid'
                        st.rerun()

# -----------------------------------------------------------------------------
# 7. 视图调度
# -----------------------------------------------------------------------------
if st.session_state.view_mode == 'quiz':
    render_quiz_ui(st.session_state.current_q_index, is_review_mode=False)

elif st.session_state.view_mode == 'review_mistakes':
    status_map = get_user_history()
    wrong_indices = [idx for idx, is_right in status_map.items() if not is_right]
    wrong_indices.sort()
    
    if not wrong_indices:
        st.balloons()
        st.success("🎉 错题本已清空！")
        if st.button("返回主页"):
            st.session_state.view_mode = 'grid'
            st.rerun()
    else:
        if st.session_state.mistake_pointer >= len(wrong_indices):
            st.session_state.mistake_pointer = 0
            
        current_wrong_q_idx = wrong_indices[st.session_state.mistake_pointer]
        
        render_quiz_ui(
            current_wrong_q_idx, 
            is_review_mode=True, 
            total_wrong_count=len(wrong_indices),
            current_wrong_pos=st.session_state.mistake_pointer
        )
