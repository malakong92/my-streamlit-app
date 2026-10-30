
import os
import streamlit as st
from openai import OpenAI

# Streamlit 앱 제목 설정
st.set_page_config(page_title="아똥이 비서 - 통신비 컨설팅", page_icon="🤖", layout="centered")

st.title("🤖 아똥이 비서: 휴대폰 & 인터넷 최적 결합 추천")
st.write("안녕하세요! 통신비 절감을 도와주는 똑똑한 비서 **아똥이**입니다. 휴대폰 기기변경, 인터넷 교체 고민을 해결해 드립니다.")

# API 키 설정 (Streamlit Cloud 환경의 Secrets 또는 로컬 환경 변수 대응)
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    # OpenAI API Key가 환경 변수에 없을 경우 사용자로부터 직접 입력 받거나 코랩 연동 키 사용
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if not api_key:
    st.warning("⚠️ OpenAI API Key를 설정해 주세요. (Sidebar 입력 또는 환경변수 설정)")
    st.stop()

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=api_key)

# 가이드 및 기본 추천 시나리오 요약 정보 제공
with st.expander("💡 아똥이 비서의 핵심 꿀팁 보기"):
    st.markdown("""
    1. **매장 인터넷 (KT로 신규 가입)**: 신규 가입 사은품(현금/상품권 지원)을 최대한 챙기세요! (보통 30~40만 원 상당 수령 가능)
    2. **휴대폰 기기변경 옵션**:
       - **옵션 A (KT 결합)**: 휴대폰을 KT로 번호이동(MNP)하여 새로 가입하는 인터넷과 결합 할인받기!
       - **옵션 B (LG 결합 유지)**: LG 기기변경을 유지해야 한다면 집 인터넷(LG)과의 결합 상태를 유지 및 강화하기!
    """)

# 세션 상태로 대화 기록 관리
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system", 
            "content": (
                "너는 친절하고 위트 있는 AI 통신 전문 비서 '아똥이'야. "
                "사용자가 휴대폰 기기변경이나 인터넷 교체를 고민할 때 가장 저렴하고 혜택이 좋은 최적의 결합 상품 정보를 비교 분석해서 추천해줘. "
                "특히 매장 인터넷 KT 신규가입 사은품 팁, KT로 번호이동 시 결합 할인(옵션 A), LG 기기변경 및 집 인터넷 결합 유지(옵션 B) 등의 구조를 친절하게 설명해줘."
            )
        },
        {
            "role": "assistant", 
            "content": "안녕하세요! 아똥이 비서입니다. 폰 기기변경이나 인터넷 변경할 때 손해보지 않도록 딱 맞는 최적의 결합 꿀팁을 비교·분석해 드릴게요. 무엇이든 물어보세요!"
        }
    ]

# 기존 대화 렌더링 (System 메시지 제외)
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# 사용자 입력 처리
if prompt := st.chat_input("예: 'LG 폰 쓰고 있는데 KT 인터넷 신규 가입하면 결합 혜택 어떻게 돼?'"):
    # 사용자 메시지 화면에 표시 및 저장
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # API 호출 및 스트리밍 효과 구현
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Chat Completion API 호출
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=st.session_state.messages,
                stream=True
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            st.error(f"에러가 발생했습니다: {e}")
            full_response = "죄송합니다, 답변을 생성하는 중 오류가 발생했습니다."
            message_placeholder.markdown(full_response)
            
    st.session_state.messages.append({"role": "assistant", "content": full_response})
