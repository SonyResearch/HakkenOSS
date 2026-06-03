import json
import os
from typing import Any

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

SUCCESS_CODE = 200
API_ENDPOINT = os.getenv("EXPLAIN_API_ENDPOINT")


def call_api(url: str, data: dict[str, Any]):
    json_data = json.dumps(data)

    headers = {"Content-Type": "application/json"}

    response = requests.post(url, data=json_data, headers=headers)

    if response.status_code == SUCCESS_CODE:
        result = response.json()
        logger.info(f"Prediction result: {result}")
        return result
    print(f"Error: {response.status_code} {response.text}")
    return None


def main():
    st.title("Simple Explainer")
    st.write("Enter the triple components and get explanatory paths")

    # Input form
    with st.form("triple_input"):
        col1, col2, col3 = st.columns(3)

        with col1:
            subject = st.text_input("Subject OCID", value="101000241533")

        with col2:
            predicate = st.text_input("Predicate", value="INHIBITS")

        with col3:
            object = st.text_input("Object OCID", value="101000106424")

        batch_size = st.slider("Batch size", min_value=1, max_value=64, value=10)

        num_explanations = st.slider("Number of explanations", min_value=1, max_value=10, value=5)

        submitted = st.form_submit_button("Get Explanations")

    if submitted:
        with st.spinner("Fetching explanations..."):
            data = {
                "triples_to_probe": [[subject, predicate, object]],
                "num_explanations": num_explanations,
                "batch_size": batch_size,
            }
            result = call_api(API_ENDPOINT, data)

            if result:
                explanation_key = f"{subject} - [{predicate}] -> {object}"

                explanations = result["explanations"][explanation_key]

                df = pd.DataFrame(explanations)

                df = df.sort_values("score", ascending=True)

                st.subheader("Explanations")

                tab1, tab2 = st.tabs(["Interactive View", "Raw Data"])

                with tab1:
                    for idx, row in df.iterrows():
                        with st.expander(f"Path {idx + 1} (Score: {row['score']:.4f})"):
                            # Split the explanation into steps
                            steps = row["data"].strip("[]").split("] <> [")
                            for hop_num, hop in enumerate(steps, 1):
                                st.write(f"Hop {hop_num}: [{hop}]")

                with tab2:
                    st.dataframe(df)

                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download results as CSV",
                    data=csv,
                    file_name="explanations.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
