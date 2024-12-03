import streamlit as st
from modules.distilbert_model import analyze_sentiment
from modules.xai import (
    explain_with_shap,
    explain_with_lime,
    get_lime_explanation_data,
    plot_lime_explanation,
)


# Function to set query parameters
def set_query_params(page):
    st.query_params.update({"page": page})


# Function to get the current query parameter value
def get_query_params():
    query_params = st.query_params
    return query_params.get("page", ["Landing Page"])[0]


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to", ["Landing Page", "Learning Modules", "Case Studies"]
    )

    if page == "Landing Page":
        st.title("Welcome to XAI Fundamentals")
        st.markdown(
            """
            This application provides an easy-to-understand introduction to Explainable AI (XAI). Learn how XAI techniques make machine learning models more transparent and interpretable. Through interactive modules, discover how AI models work, explore sentiment analysis examples, and understand the "why" behind predictions.
            """
        )

        st.subheader("What is Explainable AI (XAI)?")
        with st.expander("Click to Learn More"):
            st.write(
                """
                    Explainable AI (XAI) is a collection of methodologies, tools, and techniques aimed at making machine learning (ML) models 
                    more interpretable, transparent, and accessible to humans. In many AI applications, models function as "black boxes," 
                    generating predictions or decisions without clearly revealing the logic behind them. XAI seeks to address this challenge 
                    by providing insights into how these models operate and why they make specific predictions.

                    By bridging the gap between complex algorithms and human understanding, XAI plays a critical role in fostering trust and 
                    accountability in AI systems. It allows users, developers, and stakeholders to analyze and interpret model behavior, 
                    uncover potential biases, and ensure that the system aligns with ethical and regulatory standards.

                    The applications of XAI extend across industries:
                    - In **healthcare**, it can explain diagnostic predictions, ensuring doctors and patients understand the rationale behind 
                    a model's recommendation.
                    - In **finance**, XAI provides clarity for credit scoring or fraud detection, making the process fair and auditable.
                    - In **legal and compliance settings**, XAI helps organizations adhere to transparency regulations by providing clear 
                    explanations for automated decisions.

                    XAI methods work by shedding light on various aspects of model behavior:
                    - They identify which features (inputs) most influence predictions, helping users understand the decision-making process.
                    - They visualize patterns and relationships within the data, offering an intuitive view of model logic.
                    - They highlight inconsistencies or biases in models, empowering users to refine and improve AI systems.
                    """
            )

        st.subheader("Why is XAI Important?")
        with st.expander("Expand to Learn About Its Importance"):
            st.write(
                """
                - **Transparency:** Understanding the decision-making process of AI systems is crucial for ensuring that their predictions and decisions make sense to humans. Transparency enables users to verify whether the system is functioning as intended and to detect errors or inconsistencies in its outputs. This is particularly important in critical industries like healthcare, where understanding why a model recommended a particular diagnosis can make a significant difference in outcomes.

                - **Trustworthiness:** Building trust in AI systems requires clear explanations of their behavior. When users and stakeholders understand how a model works and why it makes certain decisions, they are more likely to adopt and rely on the technology. Trustworthy AI systems can lead to better collaboration between humans and machines, as users feel confident in the system’s reliability and fairness.

                - **Bias Detection:** Identifying and addressing biases in AI models is a fundamental aspect of creating ethical and fair AI systems. Biases in training data or algorithms can lead to discriminatory or unfair outcomes, especially when dealing with sensitive topics like hiring, lending, or healthcare. XAI helps uncover these biases by explaining which features influence decisions, allowing developers to refine models and ensure equitable treatment for all users.

                - **Compliance:** Many industries have regulatory requirements that mandate explainability for automated decision-making systems. For example, financial institutions must provide clear justifications for credit approval or denial decisions, and healthcare providers must ensure compliance with standards for patient safety. XAI tools enable organizations to meet these requirements by providing interpretable and auditable explanations of model behavior, reducing legal and reputational risks.
                """
            )

        st.subheader("Learn More About XAI")
        st.markdown(
            """
            Explore these external resources to deepen your understanding of XAI:
            - [A Beginner's Guide to XAI](https://medium.com/@okonstantinova/explainable-ai-xai-the-easy-guide-for-beginners-41e0753e03a7)
            - [SHAP Documentation](https://shap.readthedocs.io/en/latest/)
            - [LIME Overview](https://marcotcr.github.io/lime/)
            - [XAI in Practice](https://arxiv.org/abs/1902.01870)
            """
        )

        # Add interactive icons
        st.subheader("Explore Interactive Modules")
        col1, col2 = st.columns([1, 1])

        with col1:
            with st.expander("📘 Learn XAI Basics"):
                st.write(
                    """
                    The **Learning Modules** section provides an easy "try-it-yourself" function. 
                    Simply input some text into the text box, and the model will perform **sentiment analysis** 
                    to classify the sentiment as positive, neutral, or negative. 
                    To enhance transparency, XAI techniques such as **SHAP** and **LIME** can explain 
                    why the model made its prediction, breaking down the impact of each word or feature in your text.
                    """
                )

        with col2:
            with st.expander("🔍 Explore Case Studies"):
                st.write(
                    """
                    The **Case Studies** section demonstrates how XAI techniques are applied to real-world datasets. 
                    Explore examples where models have been used in areas like customer feedback analysis and healthcare. 
                    These case studies show how interpretability methods uncover biases, ensure fairness, 
                    and build trust in AI systems. Learn how XAI can drive actionable insights in practical scenarios!
                    """
                )

        st.subheader("Feedback & Suggestions")
        st.markdown(
            """
            Got feedback or suggestions? Feel free to reach out or share your ideas with me!
            """
        )

    elif page == "Learning Modules":
        st.title("Learning Modules")
        st.write(
            """
        Welcome to the Learning Modules! Here, you'll get hands-on experience with Explainable AI (XAI) 
        by exploring sentiment analysis and model interpretability. Follow the steps below to get started:
        """
        )

        st.write("### Steps to Try It Yourself:")
        st.write(
            """
            1. **Input Text:** Enter a sentence or a paragraph in the text box provided. This could be any text 
            expressing positive, neutral, or negative sentiment (e.g., "I love this product!" or "This is terrible.").
            2. **Analyze Sentiment:** Click the 'Analyze' button to see the sentiment prediction of the input text. 
            The model will classify it as Positive, Neutral, or Negative.
            3. **Explain the Prediction:** Choose an XAI method from the sidebar, such as SHAP or LIME. Click the 
            'Explain' button to view a detailed explanation of the prediction. This will help you understand 
            which words or features contributed most to the sentiment analysis.
            """
        )

        st.write("### What You'll Learn:")
        st.write(
            """
            - **How Sentiment Analysis Works:** Understand how machine learning models classify text sentiment.
            - **XAI in Action:** See how tools like SHAP and LIME provide insights into the model's decision-making process.
            - **The Role of Features:** Learn which words or phrases have the biggest impact on the model's predictions.
            """
        )

        st.info("Get started by entering your text in the box below!")

        user_input = st.text_area("Enter text to analyze sentiment", "Type here...")
        if st.button("Analyze"):
            sentiment = analyze_sentiment(user_input)
            st.write(f"Sentiment: {sentiment}")

        xai_method = st.sidebar.selectbox("Choose XAI method", ["SHAP", "LIME"])
        if st.button("Explain"):
            if xai_method == "SHAP":
                explanation_fig = explain_with_shap(user_input)
                st.pyplot(explanation_fig)
            elif xai_method == "LIME":
                exp = explain_with_lime(user_input)
                features, values = get_lime_explanation_data(exp)
                fig = plot_lime_explanation(
                    features, values
                )  # Get the Matplotlib figure
                st.pyplot(fig)  # Display the figure in Streamlit

    elif page == "Case Studies":
        st.title("Case Studies")
        st.markdown(
            """
            Explore real-world applications of Explainable AI (XAI) and see how interpretability techniques 
            uncover insights into machine learning models' behavior. These case studies highlight the use 
            of XAI in different domains to enhance trust, fairness, and transparency.
            """
        )

        st.subheader("Examples of Case Studies:")
        st.markdown(
            """
            1. **[Customer Feedback Analysis](https://trust.bizjournals.com/blog/explainable-ai-can-help-you-build-outstanding-loyal-customer-relationships)**  
            Learn how sentiment analysis models predict customer sentiment from feedback, and how tools like SHAP 
            break down which words or phrases influenced the predictions most. 

            2. **[Healthcare Diagnostics](https://pmc.ncbi.nlm.nih.gov/articles/PMC9609212/)**  
            Understand how XAI is used in healthcare to explain predictions from diagnostic models, enabling 
            doctors and patients to trust and verify the outcomes.

            3. **[Financial Risk Assessment](https://strathprints.strath.ac.uk/89573/1/FRIL-WPS-2024-Explainable-AI-for-financial-risk-management.pdf)**  
            Discover how credit scoring models leverage interpretability methods to justify loan approval decisions, 
            ensuring fairness and compliance with regulatory standards.

            4. **[Bias Detection in Recruitment](https://www.linkedin.com/pulse/exploring-horizon-explainable-ai-human-resources-selection-gupta-lztxf/)**  
            See how XAI can reveal biases in hiring models, helping organizations make fairer and more inclusive decisions.
            """
        )

        st.info(
            """
            Each case study demonstrates how XAI not only makes models more interpretable but also helps stakeholders 
            make informed, responsible decisions. Click on the links above to dive deeper into each use case and 
            see the value of XAI in action!
            """
        )

        # Assume function to display case studies


if __name__ == "__main__":
    main()
