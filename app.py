
import os
import streamlit as st
from openai import OpenAI

# OpenAI 클라이언트 생성
client = OpenAI()

# 페이지 설정
st.set_page_config(page_title="한영 번역기", page_icon="🔠")
st.title("🔠 실시간 한영 번역기")
st.write("한글 문장을 입력하시면 자연스러운 영어로 직접 번역해 드립니다.")

# 사용자 입력(텍스트)
korean_text = st.text_area(
    "번역할 한글 문장을 입력하세요.",
    height=150,
    placeholder="예: 인공지능은 우리의 일상과 업무 방식을 빠르게 변화시키고 있습니다."
)

# 번역 버튼
if st.button("영어로 번역하기"):

    if not korean_text.strip():
        st.warning("번역할 한글 텍스트를 입력해주세요!")
    else:
        with st.spinner("번역 중입니다..."):

            # 시스템 역할
            sys_role = """
            당신은 한글을 매우 정교하고 자연스럽게 영어로 바꾸어 주는 전문 번역가입니다.
            정형적이거나 직역된 번역 대신 원어민이 일상 혹은 비즈니스에서 자주 쓰는 세련된 표현으로 만들어 주세요.
            """

            # 프롬프트 구성
            prompt = f"""
            다음 문장을 영어로 번역하세요.
            TEXT:
            {korean_text}
            """

            # GPT 호출
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": sys_role},
                    {"role": "user", "content": prompt}
                ]
            )

            # 결과 출력
            translated_text = response.choices[0].message.content

            st.subheader("번역 결과")
            st.success(translated_text)
