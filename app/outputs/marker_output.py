import streamlit as st

class CareOutput:
    def __init__(self, main):
        self.main = main

    def care_home(self, info):
        st.markdown(
            f"""
                <div style="display: flex; justify-content: space-between;">
                    <strong>Care Home:</strong> <span>{info.get('name', 'N/A')}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <strong>Postcode:</strong> <span>{info.get('postcode', 'N/A')}</span>
                </div>
                """,
            unsafe_allow_html=True
        )

