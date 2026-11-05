
import streamlit as st

st.divider()

with st.expander("💡 추천 명령", expanded=True):
    st.caption(
        "이메일 초안은 내용을 미리 보여줍니다. "
        "이메일 발송은 아래 수신자에게 즉시 발송합니다."
    )

    quick_to = st.text_input(
        "이메일 수신자",
        key="quick_email_to",
        placeholder="example@gmail.com"
    )

    quick_subject = st.text_input(
        "이메일 제목",
        value="회의 핵심 요약 및 담당 업무 공유",
        key="quick_email_subject"
    )

    quick_commands = [
        ("🎤 회의 핵심 요약", "summary"),
        ("📋 내 담당 업무", "tasks"),
        ("✅ 최종 결정사항", "decisions"),
        ("📝 이메일 초안", "draft"),
        ("📧 이메일 발송", "send"),
        ("🔎 업무 근거 확인", "evidence")
    ]

    # 기존 코드에서 관리하는 회의록을 읽습니다.
    # 실제 회의록 변수명이 다르면 이 부분을 변경하세요.
    quick_source = (
        st.session_state.get("meeting_text")
        or meeting_text
        or ""
    ).strip()

    selected_action = None

    c1, c2 = st.columns(2)

    for i, (label, action) in enumerate(quick_commands):
        with (c1 if i % 2 == 0 else c2):
            if st.button(
                label,
                key=f"quick_{action}",
                use_container_width=True
            ):
                selected_action = action

    if selected_action:
        if not quick_source:
            st.warning("먼저 회의록을 업로드하거나 입력해주세요.")

        elif selected_action == "send" and not quick_to.strip():
            st.warning("이메일 수신자를 입력해주세요.")

        elif selected_action == "send" and not quick_subject.strip():
            st.warning("이메일 제목을 입력해주세요.")

        else:
            try:
                with st.spinner("회의록을 분석하고 근거를 확인하고 있습니다."):
                    quick_data = analyze_quick_meeting(quick_source)

                # 결과를 저장해 다른 버튼 클릭이나 화면 갱신 후에도 유지
                st.session_state.quick_result = {
                    "source": quick_source,
                    "action": selected_action,
                    "data": quick_data
                }

                if selected_action in ("draft", "send"):
                    email_body = build_quick_email(quick_data)

                    st.session_state.quick_email_draft = {
                        "source": quick_source,
                        "to": quick_to.strip(),
                        "subject": quick_subject.strip(),
                        "body": email_body
                    }

                if selected_action == "send":
                    draft = st.session_state.quick_email_draft

                    # 기존 이메일 발송 함수 사용
                    # 버튼 클릭 이벤트 안에서만 호출하므로,
                    # 일반적인 화면 재실행으로 재발송되지 않습니다.
                    with st.spinner("이메일을 발송하고 있습니다."):
                        send_result = send_email(
                            to=draft["to"],
                            subject=draft["subject"],
                            body=draft["body"]
                        )

                    # 반환값의 성공/실패 형식은 기존 함수에 따라 다르므로
                    # 무조건 '발송 성공'으로 표시하지 않습니다.
                    st.info("이메일 발송 함수를 실행했습니다. 결과를 확인해주세요.")

                    if send_result is not None:
                        st.write(send_result)

            except Exception as e:
                st.error(
                    f"명령 실행 오류: {type(e).__name__} — {e}"
                )

    # =====================================================
    # 저장된 결과 표시
    # =====================================================

    saved_result = st.session_state.get("quick_result")

    if saved_result and saved_result["source"] == quick_source:
        action = saved_result["action"]

        if action in ("draft", "send"):
            draft = st.session_state.get("quick_email_draft")

            if draft and draft["source"] == quick_source:
                st.subheader("📝 이메일 초안")
                st.write("수신자:", draft["to"] or "미입력")
                st.write("제목:", draft["subject"])
                st.text(draft["body"])

                with st.expander("이메일 작성에 사용한 JSON"):
                    st.json(saved_result["data"])

        else:
            show_quick_result(
                action=action,
                data=saved_result["data"],
                source=quick_source
            )
# 11. AI 분석 결과
# =========================================================

if st.session_state.analysis:
    with st.expander("🔎 구조화 분석 결과"):
        st.json(
            st.session_state.analysis
        )
        st.download_button(
            "마지막 분석 JSON 다운로드",
            json.dumps(
                st.session_state.analysis,
                ensure_ascii=False,
                indent=2
            ),
            file_name="aivle_analysis.json"
        )

st.caption(
    "에이블 · 회의록 분석 / STT / TTS / "
    "Tool Calling / 최근 10개 대화"
)
