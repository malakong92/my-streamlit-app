
import streamlit as st
import sqlite3
import tempfile
import os
import secrets
import hashlib
import hmac
from pathlib import Path
from contextlib import closing
from operator import itemgetter

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage


# ============================================================
# 1. 기본 설정
# ============================================================

st.set_page_config(
    page_title="AIVLE School 학습 도우미",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)

CHROMA_DIR = "chroma_db"
COLLECTION = "aivle_docs"

BOARD_DB_PATH = (
    Path(tempfile.gettempdir()) / "aivle_senior_board.sqlite3"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "app_view" not in st.session_state:
    st.session_state.app_view = "chat"

if "admin_login" not in st.session_state:
    st.session_state.admin_login = False


# ============================================================
# 2. 기존 RAG 기능 (설정 유지)
# ============================================================

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@st.cache_resource
def get_retriever():
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embedding,
        persist_directory=CHROMA_DIR
    )

    return vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )


@st.cache_resource
def get_chain():
    retriever = get_retriever()

    s_msg = (
        "당신은 주어진 문서를 기반으로 답변하는 "
        "AIVLE School 학습 도우미입니다.\n\n"
        "[문서]\n\n{context}"
    )

    h_msg = "{input}"

    messages = [
        ("system", s_msg),
        ("human", h_msg)
    ]

    prompt = ChatPromptTemplate.from_messages(messages)

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        streaming=True,
        temperature=0
    )

    rag_chain = (
        {
            "context": (
                itemgetter("input")
                | retriever
                | format_docs
            ),
            "input": itemgetter("input")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ============================================================
# 3. SQLite 데이터베이스
# ============================================================

def board_connect():
    conn = sqlite3.connect(
        str(BOARD_DB_PATH),
        timeout=15
    )

    conn.execute("PRAGMA busy_timeout=15000")
    conn.execute("PRAGMA foreign_keys=ON")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            author TEXT NOT NULL,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            created_at TEXT NOT NULL
                DEFAULT (datetime('now','localtime'))
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS replies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id INTEGER NOT NULL,
            author TEXT NOT NULL,
            body TEXT NOT NULL,
            created_at TEXT NOT NULL
                DEFAULT (datetime('now','localtime')),
            FOREIGN KEY(question_id)
                REFERENCES questions(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS mentors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            generation TEXT NOT NULL,
            track TEXT NOT NULL,
            company TEXT NOT NULL,
            job TEXT NOT NULL,
            introduction TEXT NOT NULL,
            email TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            reply_code_hash TEXT,
            created_at TEXT NOT NULL
                DEFAULT (datetime('now','localtime'))
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS mentoring_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mentor_id INTEGER NOT NULL,
            junior_name TEXT NOT NULL,
            junior_email TEXT NOT NULL,
            topic TEXT NOT NULL,
            message TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL
                DEFAULT (datetime('now','localtime')),
            FOREIGN KEY(mentor_id)
                REFERENCES mentors(id)
        )
    """)

    conn.commit()
    return conn


def db_query(sql, params=()):
    with closing(board_connect()) as conn:
        return conn.execute(sql, params).fetchall()


def db_execute(sql, params=()):
    with closing(board_connect()) as conn:
        cursor = conn.execute(sql, params)
        conn.commit()
        return cursor.rowcount


# ============================================================
# 4. 관리자 / 선배 인증
# ============================================================

def get_admin_password():
    try:
        password = st.secrets.get(
            "BOARD_ADMIN_PASSWORD", ""
        )
        if password:
            return str(password)
    except Exception:
        pass

    return os.environ.get(
        "BOARD_ADMIN_PASSWORD", ""
    )


def hash_code(code):
    salt = secrets.token_bytes(16)

    digest = hashlib.pbkdf2_hmac(
        "sha256",
        code.encode("utf-8"),
        salt,
        200000
    )

    return salt.hex() + ":" + digest.hex()


def verify_code(code, saved):
    try:
        salt_hex, digest_hex = saved.split(":")

        digest = hashlib.pbkdf2_hmac(
            "sha256",
            code.encode("utf-8"),
            bytes.fromhex(salt_hex),
            200000
        )

        return hmac.compare_digest(
            digest,
            bytes.fromhex(digest_hex)
        )
    except (ValueError, AttributeError, TypeError):
        return False


def verify_senior(mentor_id, code):
    rows = db_query("""
        SELECT reply_code_hash
        FROM mentors
        WHERE id = ?
          AND status = 'approved'
    """, (mentor_id,))

    if not rows or not rows[0][0]:
        return False

    return verify_code(code, rows[0][0])


def valid_email(email):
    email = email.strip()
    return (
        "@" in email
        and "." in email.split("@")[-1]
        and " " not in email
    )


# ============================================================
# 5. 사이드바
# ============================================================

with st.sidebar:

    st.markdown("## 📘 AIVLE")

    st.divider()

    st.markdown("### 📚 AI 학습 도우미")

    if st.button(
        "🏠 AI 학습 도우미",
        use_container_width=True,
        key="sidebar_chat"
    ):
        st.session_state.app_view = "chat"
        st.rerun()

    if st.button(
        "🗑️ 대화 삭제",
        use_container_width=True,
        key="sidebar_clear"
    ):
        st.session_state.messages = []
        st.session_state.app_view = "chat"
        st.rerun()

    st.divider()

    # 요청한 위치: 사이드바 내부
    st.markdown("### 🎓 선배들에게 질문하기!")

    if st.button(
        "💬 선배 Q&A",
        use_container_width=True,
        key="sidebar_qna"
    ):
        st.session_state.app_view = "qna"
        st.rerun()

    if st.button(
        "🤝 선배 멘토",
        use_container_width=True,
        key="sidebar_mentor"
    ):
        st.session_state.app_view = "mentor"
        st.rerun()

    st.caption(
        "졸업 선배에게 질문하거나 "
        "멘토링을 신청해 보세요."
    )

    st.divider()

    with st.expander("📖 백서 검색 안내"):
        st.write(
            "기존 Chroma DB에 저장된 "
            "AIVLE School 문서를 검색합니다."
        )

    with st.expander("⚙️ 운영자"):
        if st.button(
            "운영자 관리 페이지",
            use_container_width=True,
            key="sidebar_admin"
        ):
            st.session_state.app_view = "admin"
            st.rerun()


# ============================================================
# 6. AI 학습 도우미 화면
# ============================================================

def render_chat():
    st.title("📘 AIVLE School 학습 도우미")

    st.caption(
        "AI가 AIVLE School 백서와 문서를 "
        "기반으로 질문에 답변합니다."
    )

    st.divider()

    # 기존 대화 출력
    for message in st.session_state.messages:
        role = (
            "user"
            if isinstance(message, HumanMessage)
            else "assistant"
        )

        with st.chat_message(role):
            st.markdown(message.content)

    # 기존 RAG 사용자 입력
    user_input = st.chat_input(
        "메시지를 입력하세요.",
        key="rag_chat_input"
    )

    if user_input:
        st.chat_message("user").markdown(user_input)

        st.session_state.messages.append(
            HumanMessage(content=user_input)
        )

        with st.chat_message("assistant"):
            try:
                chain = get_chain()

                response_stream = chain.stream({
                    "input": user_input
                })

                full_response = st.write_stream(
                    response_stream
                )

                st.session_state.messages.append(
                    AIMessage(content=full_response)
                )

            except Exception:
                st.error(
                    "답변 생성 중 오류가 발생했습니다. "
                    "API 키와 Chroma DB를 확인해 주세요."
                )


# ============================================================
# 7. 선배 Q&A 화면
# ============================================================

def render_qna():
    st.title("💬 선배 Q&A")

    st.write(
        "에이블스쿨 선배에게 궁금한 점을 "
        "질문해 보세요."
    )

    st.caption(
        "질문은 공개되며, 관리자 인증을 받은 "
        "선배만 답변을 등록할 수 있습니다."
    )

    st.divider()

    # 질문 작성
    with st.expander(
        "✍️ 새로운 질문 작성",
        expanded=False
    ):

        with st.form(
            "question_form",
            clear_on_submit=True
        ):

            author = st.text_input(
                "이름 또는 닉네임",
                max_chars=30,
                placeholder="예: 10기 교육생"
            )

            title = st.text_input(
                "질문 제목",
                max_chars=120,
                placeholder=(
                    "예: DX 프로젝트는 "
                    "어떻게 준비하셨나요?"
                )
            )

            body = st.text_area(
                "질문 내용",
                max_chars=4000,
                height=150
            )

            submitted = st.form_submit_button(
                "질문 등록",
                use_container_width=True
            )

        if submitted:
            if not title.strip() or not body.strip():
                st.warning(
                    "제목과 질문 내용을 입력해 주세요."
                )
            else:
                db_execute("""
                    INSERT INTO questions
                    (author, title, body)
                    VALUES (?, ?, ?)
                """, (
                    author.strip() or "익명 후배",
                    title.strip(),
                    body.strip()
                ))

                st.rerun()

    # 질문 검색
    st.markdown("### 📋 질문 목록")

    search = st.text_input(
        "질문 검색",
        placeholder="검색어를 입력하세요.",
        key="qna_search"
    )

    questions = db_query("""
        SELECT id, author, title, body, created_at
        FROM questions
        ORDER BY id DESC
    """)

    if search.strip():
        keyword = search.strip().lower()

        questions = [
            q for q in questions
            if keyword in (
                q[2] + " " + q[3]
            ).lower()
        ]

    if not questions:
        st.info(
            "아직 질문이 없습니다. "
            "첫 질문을 등록해 보세요."
        )
        return

    # 인증 완료된 선배 목록
    approved_seniors = db_query("""
        SELECT id, name, generation, company
        FROM mentors
        WHERE status = 'approved'
        ORDER BY id DESC
    """)

    senior_options = {
        f"{name} · {generation} · {company} (#{sid})": sid
        for sid, name, generation, company
        in approved_seniors
    }

    for qid, author, title, body, created in questions:

        replies = db_query("""
            SELECT author, body, created_at
            FROM replies
            WHERE question_id = ?
            ORDER BY id ASC
        """, (qid,))

        status = (
            "✅ 답변 완료"
            if replies
            else "⏳ 답변 대기"
        )

        with st.expander(
            f"{status} | {title} | 답변 {len(replies)}개"
        ):

            st.caption(
                f"👤 {author} · {created}"
            )

            st.write(body)
            st.divider()

            # 선배 답변 표시
            if replies:
                st.markdown("#### 🎓 선배 답변")

                for reply_author, reply_body, reply_date in replies:
                    st.markdown(
                        f"**✅ {reply_author} 선배**"
                    )

                    st.caption(reply_date)
                    st.write(reply_body)
                    st.divider()

            else:
                st.caption(
                    "아직 등록된 답변이 없습니다."
                )

            # 인증 선배만 답변 등록
            st.markdown("#### ✏️ 선배 답변 작성")

            if not senior_options:
                st.info(
                    "현재 인증된 선배가 없습니다."
                )
                continue

            with st.form(
                f"reply_form_{qid}",
                clear_on_submit=True
            ):

                selected_senior = st.selectbox(
                    "인증된 선배 선택",
                    options=list(senior_options.keys()),
                    key=f"senior_{qid}"
                )

                code = st.text_input(
                    "선배 인증 코드",
                    type="password",
                    key=f"code_{qid}"
                )

                reply = st.text_area(
                    "답변 내용",
                    max_chars=4000,
                    key=f"reply_{qid}"
                )

                reply_submit = st.form_submit_button(
                    "답변 등록"
                )

            if reply_submit:
                senior_id = senior_options[
                    selected_senior
                ]

                if not reply.strip():
                    st.warning(
                        "답변을 입력해 주세요."
                    )

                elif not verify_senior(
                    senior_id, code
                ):
                    st.error(
                        "선배 인증 코드가 일치하지 않습니다."
                    )

                else:
                    senior_name = db_query("""
                        SELECT name
                        FROM mentors
                        WHERE id = ?
                    """, (senior_id,))[0][0]

                    db_execute("""
                        INSERT INTO replies
                        (question_id, author, body)
                        VALUES (?, ?, ?)
                    """, (
                        qid,
                        senior_name,
                        reply.strip()
                    ))

                    st.rerun()


# ============================================================
# 8. 선배 멘토 화면
# ============================================================

def render_mentor():
    st.title("🤝 선배 멘토")

    st.write(
        "에이블스쿨을 졸업하고 취업한 "
        "선배에게 멘토링을 신청해 보세요."
    )

    st.divider()

    # 멘토 등록 신청
    with st.expander(
        "🎓 졸업 선배이신가요? 멘토 등록하기"
    ):

        st.caption(
            "관리자가 졸업 및 재직 여부를 "
            "별도로 확인한 후 등록됩니다."
        )

        with st.form(
            "mentor_registration",
            clear_on_submit=True
        ):

            name = st.text_input(
                "이름 또는 닉네임",
                max_chars=30
            )

            generation = st.text_input(
                "에이블스쿨 기수",
                placeholder="예: 6기",
                max_chars=20
            )

            track = st.selectbox(
                "교육 트랙",
                [
                    "DX 컨설턴트",
                    "AI 개발자",
                    "기타"
                ]
            )

            company = st.text_input(
                "재직 회사",
                max_chars=100
            )

            job = st.text_input(
                "담당 직무",
                max_chars=100
            )

            introduction = st.text_area(
                "자기소개 및 멘토링 가능 분야",
                max_chars=1000
            )

            email = st.text_input(
                "관리자 확인용 이메일",
                max_chars=200
            )

            submitted = st.form_submit_button(
                "멘토 등록 신청",
                use_container_width=True
            )

        if submitted:
            if not all([
                name.strip(),
                generation.strip(),
                company.strip(),
                job.strip(),
                introduction.strip(),
                email.strip()
            ]):
                st.warning(
                    "모든 항목을 입력해 주세요."
                )

            elif not valid_email(email):
                st.warning(
                    "올바른 이메일을 입력해 주세요."
                )

            else:
                db_execute("""
                    INSERT INTO mentors
                    (
                        name, generation, track,
                        company, job,
                        introduction, email
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    name.strip(),
                    generation.strip(),
                    track,
                    company.strip(),
                    job.strip(),
                    introduction.strip(),
                    email.strip()
                ))

                st.success(
                    "멘토 등록 신청이 완료되었습니다. "
                    "관리자 승인 후 목록에 표시됩니다."
                )

    # 멘토 찾기
    st.markdown("### 🔎 나에게 맞는 멘토 찾기")

    col1, col2 = st.columns([1, 2])

    with col1:
        job_filter = st.selectbox(
            "직무 선택",
            [
                "전체",
                "DX 컨설팅",
                "데이터 분석",
                "AI 개발",
                "기타"
            ]
        )

    with col2:
        search = st.text_input(
            "멘토 검색",
            placeholder="회사, 직무, 이름 검색"
        )

    mentors = db_query("""
        SELECT
            id, name, generation,
            track, company, job, introduction
        FROM mentors
        WHERE status = 'approved'
        ORDER BY id DESC
    """)

    # 직무 필터
    def job_match(mentor):
        text = (
            str(mentor[3]) + " " +
            str(mentor[5])
        ).lower()

        if job_filter == "전체":
            return True

        if job_filter == "DX 컨설팅":
            return (
                "dx" in text
                or "컨설" in text
            )

        if job_filter == "데이터 분석":
            return (
                "데이터" in text
                or "분석" in text
            )

        if job_filter == "AI 개발":
            return (
                "ai" in text
                or "개발" in text
            )

        return True

    mentors = [
        m for m in mentors
        if job_match(m)
    ]

    if search.strip():
        keyword = search.strip().lower()

        mentors = [
            m for m in mentors
            if keyword in " ".join(
                str(x or "")
                for x in m[1:]
            ).lower()
        ]

    if not mentors:
        st.info(
            "현재 등록된 인증 멘토가 없습니다."
        )
        return

    st.caption(
        f"검색된 멘토 {len(mentors)}명"
    )

    # 멘토 프로필 카드
    for (
        mid, name, generation,
        track, company, job, intro
    ) in mentors:

        with st.container(border=True):

            st.markdown(
                f"### 🎓 {name} 선배"
            )

            st.caption(
                f"✅ 관리자 인증 · "
                f"{generation} · {track}"
            )

            st.write(
                f"🏢 **회사:** {company}"
            )

            st.write(
                f"💼 **직무:** {job}"
            )

            st.write(intro)

            # 멘토링 신청
            with st.expander(
                "✉️ 멘토링 신청하기"
            ):

                with st.form(
                    f"mentor_request_{mid}",
                    clear_on_submit=True
                ):

                    junior = st.text_input(
                        "신청자 이름 또는 닉네임",
                        max_chars=30,
                        key=f"junior_{mid}"
                    )

                    junior_email = st.text_input(
                        "연락받을 이메일",
                        max_chars=200,
                        key=f"junior_email_{mid}"
                    )

                    topic = st.selectbox(
                        "멘토링 주제",
                        [
                            "취업 및 진로",
                            "포트폴리오",
                            "면접 준비",
                            "프로젝트",
                            "직무 상담",
                            "기타"
                        ],
                        key=f"topic_{mid}"
                    )

                    message = st.text_area(
                        "상담하고 싶은 내용",
                        max_chars=2000,
                        key=f"message_{mid}"
                    )

                    request_submit = st.form_submit_button(
                        "멘토링 신청",
                        use_container_width=True
                    )

                if request_submit:
                    if not all([
                        junior.strip(),
                        junior_email.strip(),
                        message.strip()
                    ]):
                        st.warning(
                            "모든 항목을 입력해 주세요."
                        )

                    elif not valid_email(
                        junior_email
                    ):
                        st.warning(
                            "이메일 형식을 확인해 주세요."
                        )

                    else:
                        db_execute("""
                            INSERT INTO mentoring_requests
                            (
                                mentor_id,
                                junior_name,
                                junior_email,
                                topic,
                                message
                            )
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            mid,
                            junior.strip(),
                            junior_email.strip(),
                            topic,
                            message.strip()
                        ))

                        st.success(
                            "멘토링 신청이 접수되었습니다. "
                            "운영자가 신청 내용을 확인합니다."
                        )


# ============================================================
# 9. 운영자 페이지
# ============================================================

def render_admin():
    st.title("🔐 선배 연결 운영자")

    admin_password = get_admin_password()

    if not admin_password:
        st.error(
            "관리자 비밀번호가 설정되지 않았습니다. "
            "BOARD_ADMIN_PASSWORD를 설정해 주세요."
        )
        return

    if not st.session_state.admin_login:

        with st.form("admin_login_form"):
            password = st.text_input(
                "관리자 비밀번호",
                type="password"
            )

            login = st.form_submit_button(
                "로그인"
            )

        if login:
            if hmac.compare_digest(
                password, admin_password
            ):
                st.session_state.admin_login = True
                st.rerun()

            else:
                st.error(
                    "비밀번호가 올바르지 않습니다."
                )

        return

    if st.button("로그아웃"):
        st.session_state.admin_login = False
        st.rerun()

    tab1, tab2 = st.tabs([
        "🎓 선배 등록 승인",
        "📩 멘토링 신청 관리"
    ])

    with tab1:
        st.markdown("### 승인 대기 선배")

        pending = db_query("""
            SELECT
                id, name, generation,
                track, company, job,
                introduction, email
            FROM mentors
            WHERE status = 'pending'
            ORDER BY id ASC
        """)

        if not pending:
            st.info(
                "승인 대기 중인 선배가 없습니다."
            )

        for (
            mid, name, generation,
            track, company, job,
            intro, email
        ) in pending:

            with st.container(border=True):

                st.markdown(
                    f"**{name} · {generation}**"
                )

                st.write(
                    f"트랙: {track}"
                )

                st.write(
                    f"회사: {company}"
                )

                st.write(
                    f"직무: {job}"
                )

                st.write(
                    f"소개: {intro}"
                )

                st.write(
                    f"이메일: {email}"
                )

                st.warning(
                    "졸업 및 재직 여부를 "
                    "별도 경로로 확인한 뒤 승인하세요."
                )

                col1, col2 = st.columns(2)

                with col1:
                    if st.button(
                        "✅ 승인",
                        key=f"approve_{mid}",
                        use_container_width=True
                    ):
                        code = secrets.token_urlsafe(18)

                        db_execute("""
                            UPDATE mentors
                            SET status = 'approved',
                                reply_code_hash = ?
                            WHERE id = ?
                              AND status = 'pending'
                        """, (
                            hash_code(code),
                            mid
                        ))

                        st.session_state.new_code = (
                            name, code
                        )

                        st.rerun()

                with col2:
                    if st.button(
                        "❌ 거절",
                        key=f"reject_{mid}",
                        use_container_width=True
                    ):
                        db_execute("""
                            UPDATE mentors
                            SET status = 'rejected'
                            WHERE id = ?
                              AND status = 'pending'
                        """, (mid,))

                        st.rerun()

        if "new_code" in st.session_state:
            name, code = st.session_state.new_code

            st.success(
                f"{name} 선배 승인 완료"
            )

            st.warning(
                "아래 인증 코드는 지금만 표시됩니다. "
                "승인한 선배에게 별도로 전달하세요."
            )

            st.code(code)

            if st.button("코드 확인 완료"):
                del st.session_state.new_code
                st.rerun()

    with tab2:
        st.markdown("### 멘토링 신청 내역")

        requests = db_query("""
            SELECT
                r.id,
                m.name,
                m.email,
                r.junior_name,
                r.junior_email,
                r.topic,
                r.message,
                r.status,
                r.created_at
            FROM mentoring_requests r
            JOIN mentors m
                ON r.mentor_id = m.id
            ORDER BY r.id DESC
        """)

        if not requests:
            st.info(
                "아직 멘토링 신청이 없습니다."
            )

        for (
            rid, mentor_name, mentor_email,
            junior, junior_email,
            topic, message, status, created
        ) in requests:

            with st.expander(
                f"{topic} | {junior} → "
                f"{mentor_name} | {status}"
            ):

                st.write(
                    f"멘토: {mentor_name}"
                )

                st.write(
                    f"멘토 이메일: {mentor_email}"
                )

                st.write(
                    f"신청자: {junior}"
                )

                st.write(
                    f"신청자 이메일: {junior_email}"
                )

                st.write(
                    f"상담 내용: {message}"
                )

                st.caption(
                    f"신청일: {created}"
                )

                status_map = {
                    "pending": "접수",
                    "contacted": "연락 완료",
                    "completed": "멘토링 완료"
                }

                labels = list(status_map.values())

                reverse_map = {
                    value: key
                    for key, value
                    in status_map.items()
                }

                current = status_map.get(
                    status, "접수"
                )

                selected = st.selectbox(
                    "진행 상태",
                    labels,
                    index=labels.index(current),
                    key=f"status_{rid}"
                )

                if st.button(
                    "상태 저장",
                    key=f"save_{rid}"
                ):
                    db_execute("""
                        UPDATE mentoring_requests
                        SET status = ?
                        WHERE id = ?
                    """, (
                        reverse_map[selected],
                        rid
                    ))

                    st.rerun()


# ============================================================
# 10. 메인 화면 전환
# ============================================================

try:
    # 데이터베이스 테이블 생성
    with closing(board_connect()):
        pass

    view = st.session_state.app_view

    if view == "chat":
        render_chat()

    elif view == "qna":
        render_qna()

    elif view == "mentor":
        render_mentor()

    elif view == "admin":
        render_admin()

    else:
        st.session_state.app_view = "chat"
        st.rerun()

except sqlite3.Error:
    st.error(
        "게시판 데이터베이스 오류가 발생했습니다."
    )
