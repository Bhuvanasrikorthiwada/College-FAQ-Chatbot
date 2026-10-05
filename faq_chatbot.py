import math
import re
import time
from collections import Counter

import streamlit as st


# -----------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------

st.set_page_config(
    page_title="College FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 College FAQ Chatbot")
st.write("Ask me anything about the college!")


# -----------------------------------------
# FAQ DATA
# -----------------------------------------

faqs = [
    {
        "question": "What is the admission process?",
        "answer": "The admission process usually includes registration, submitting the required documents, and completing the admission formalities."
    },
    {
        "question": "What documents are required for admission?",
        "answer": "Commonly required documents include academic certificates, identification proof, photographs, and other documents specified by the institution."
    },
    {
        "question": "How can I pay the fees?",
        "answer": "Fees can be paid using the payment methods provided by the institution, such as online payment or bank transfer."
    },
    {
        "question": "How can I contact support?",
        "answer": "You can contact the support team through the official contact details provided by the organization."
    },
    {
        "question": "What are the working hours?",
        "answer": "Working hours depend on the organization. Please check the official schedule for the latest timings."
    },
    {
        "question": "How can I reset my password?",
        "answer": "Use the 'Forgot Password' option on the login page and follow the instructions to reset your password."
    },
    {
        "question": "Can I update my personal information?",
        "answer": "Yes. If profile editing is enabled, you can update your information from your account or profile settings."
    },
    {
        "question": "How can I submit a complaint?",
        "answer": "You can submit a complaint through the organization's official support or complaint portal."
    },
    {
        "question": "What are the college working hours?",
        "answer": "The college working hours are from 9:00 AM to 4:00 PM."
    },
    {
        "question": "How can I apply for leave?",
        "answer": "Students can apply for leave through the college leave application process."
    },
    {
        "question": "How can I contact the college office?",
        "answer": "You can contact the college office during working hours for assistance."
    },
    {
        "question": "Where can I pay my college fees?",
        "answer": "College fees can be paid through the official fee payment system."
    },
    {
        "question": "How can I get my examination timetable?",
        "answer": "The examination timetable is generally provided through the official college communication channels."
    },
    {
        "question": "What should I do if I lose my ID card?",
        "answer": "Report the lost ID card to the concerned college office and follow the replacement procedure."
    }
]


# -----------------------------------------
# QUESTIONS
# -----------------------------------------

questions = [
    faq["question"]
    for faq in faqs
]


# -----------------------------------------
# STOP WORDS
# -----------------------------------------

stop_words = {
    "a", "an", "the", "is", "are", "what",
    "how", "can", "i", "to", "for", "my",
    "do", "you", "of", "and", "in", "on"
}


# -----------------------------------------
# TOKENIZATION
# -----------------------------------------

def tokenize(text):

    return [
        token
        for token in re.findall(
            r"[a-z0-9]+",
            text.lower()
        )
        if token not in stop_words
    ]


# -----------------------------------------
# TF-IDF
# -----------------------------------------

faq_tokens = [
    tokenize(question)
    for question in questions
]

document_frequency = Counter(
    token
    for tokens in faq_tokens
    for token in set(tokens)
)

inverse_document_frequency = {
    token: math.log(
        (1 + len(questions)) /
        (1 + frequency)
    ) + 1

    for token, frequency
    in document_frequency.items()
}


def vectorize(tokens):

    return {
        token: count * inverse_document_frequency[token]

        for token, count
        in Counter(tokens).items()

        if token in inverse_document_frequency
    }


def cosine_similarity(left, right):

    dot_product = sum(
        weight * right.get(token, 0)
        for token, weight in left.items()
    )

    left_norm = math.sqrt(
        sum(
            weight * weight
            for weight in left.values()
        )
    )

    right_norm = math.sqrt(
        sum(
            weight * weight
            for weight in right.values()
        )
    )

    if not left_norm or not right_norm:
        return 0.0

    return dot_product / (
        left_norm * right_norm
    )


faq_vectors = [
    vectorize(tokens)
    for tokens in faq_tokens
]


# -----------------------------------------
# CHAT HISTORY
# -----------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


# -----------------------------------------
# SUGGESTED QUESTIONS
# -----------------------------------------

st.subheader("💡 Suggested Questions")

suggestions = [
    "What is the admission process?",
    "What are the college working hours?",
    "How can I pay the fees?",
    "How can I apply for leave?"
]

cols = st.columns(2)

for i, suggestion in enumerate(suggestions):

    if cols[i % 2].button(
        suggestion,
        use_container_width=True
    ):

        st.session_state.selected_question = suggestion


# -----------------------------------------
# USER INPUT
# -----------------------------------------

question = st.chat_input(
    "Ask your question..."
)


# Use suggested question if selected

if "selected_question" in st.session_state:

    if st.session_state.selected_question:

        question = st.session_state.selected_question

        st.session_state.selected_question = None


# -----------------------------------------
# PROCESS QUESTION
# -----------------------------------------

if question:

    start_time = time.time()

    # -------------------------------------
    # DISPLAY USER QUESTION
    # -------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.write(question)

    text = question.lower().strip()

    # -------------------------------------
    # GREETINGS
    # -------------------------------------

    greetings = [
        "hello",
        "hi",
        "hey",
        "hii",
        "hiii",
        "hello there",
        "heyy"
    ]

    if text in greetings:

        response = (
            "Hello! 👋 Welcome to the "
            "College FAQ Chatbot. "
            "What would you like to know?"
        )

        matched = False
        matched_question = ""
        confidence = 1.0

    else:

        # ---------------------------------
        # FIND BEST FAQ
        # ---------------------------------

        question_vector = vectorize(
            tokenize(question)
        )

        similarities = [
            cosine_similarity(
                question_vector,
                faq_vector
            )
            for faq_vector in faq_vectors
        ]

        best_index = max(
            range(len(similarities)),
            key=similarities.__getitem__
        )

        confidence = similarities[best_index]

        matched = confidence >= 0.15

        matched_question = questions[
            best_index
        ]

        if matched:

            response = faqs[
                best_index
            ]["answer"]

        else:

            response = (
                "Sorry, I couldn't find a relevant "
                "answer in my FAQ database. "
                "Please try asking your question "
                "differently."
            )

    # -------------------------------------
    # RESPONSE TIME
    # -------------------------------------

    response_time = time.time() - start_time

    # -------------------------------------
    # SAVE RESPONSE
    # -------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

    # -------------------------------------
    # DISPLAY RESPONSE
    # -------------------------------------

    with st.chat_message("assistant"):

        st.write(response)

        # Show additional information

        if matched:

            with st.expander("🔍 Answer Details"):

                st.write(
                    "📌 Matched FAQ:"
                )

                st.write(
                    matched_question
                )

                st.progress(
                    min(confidence, 1.0)
                )

                st.write(
                    f"🎯 Match confidence: "
                    f"{confidence * 100:.1f}%"
                )

                st.write(
                    f"⚡ Response time: "
                    f"{response_time:.3f} seconds"
                )


# -----------------------------------------
# SIDEBAR
# -----------------------------------------

with st.sidebar:

    st.header("📚 FAQ Information")

    st.write(
        f"Total FAQs: {len(faqs)}"
    )

    st.write(
        "🔎 Search: TF-IDF"
    )

    st.write(
        "🎯 Matching: Cosine Similarity"
    )

    st.write(
        "💬 Chat History: Enabled"
    )

    st.divider()

    st.subheader("⚙️ Options")

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()
