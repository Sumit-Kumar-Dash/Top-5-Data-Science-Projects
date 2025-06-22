# 🚀 Top 5 Data Science Projects to Land Your First Job

> A curated list of **5 industry-ready Data Science projects** every beginner and job aspirant should have in their portfolio. Each project is selected for its **real-world impact, skill showcase, and relevance** to 2025 hiring trends — including **Generative AI** integrations!

[![YouTube Video](https://img.shields.io/badge/YouTube-Watch%20Video-red?style=for-the-badge&logo=youtube)](https://www.youtube.com/channel/UCbXwEhVG0Hv3DtdORQLs_FQ)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/sumit-kumar-dash-315378140/)

---

## 📋 Table of Contents

- [🎯 Why These Projects?](#-why-these-projects)
- [🔥 Project Showcase](#-project-showcase)
  - [1. Customer Churn Prediction](#1-customer-churn-prediction)
  - [2. GenAI Document Intelligence System](#2-genai-document-intelligence-system)
  - [3. Time Series Forecasting Pipeline](#3-time-series-forecasting-pipeline)
  - [4. E-commerce Product Reviews Scraping & Analysis](#4-e-commerce-product-reviews-scraping--analysis)
  - [5. Healthcare Analytics: Disease Risk Prediction](#5-healthcare-analytics-disease-risk-prediction)
---

## 🧠 Why These Projects?

These projects are specifically chosen to demonstrate skills that hiring managers actively seek:

✅ **Real-world Business Focus** - Each project solves real business problems  
✅ **End-to-End Implementation** - Covers end-to-end pipeline -> From data collection to deployment  
✅ **Modern Tech Stack** - Current industry tools and frameworks  
✅ **Production-Ready Code** - Clean, documented, and scalable  
✅ **Diverse Skill Set** - Covers ML, Data Analysis, Time Series, NLP, and Gen AI  
✅ **Each project** = 1 interview story 📈


---

## 🔥 Project Showcase

### 1. Customer Churn Prediction
> **Predicting customer churn to reduce revenue loss and improve retention strategies**

![Churn Dashboard](https://github.com/Sumit-Kumar-Dash/Top-5-Data-Science-Projects/blob/main/customer_churn_dashboard.png)

**📉 Business Problem:** Telecommunications companies lose millions annually due to customer churn. Acquiring new customers costs 5x more than retaining existing ones. This project identifies at-risk customers before they leave.

**🎯 Target Metrics:**
- **Recall > 75%** for high-value customers (minimize false negatives)
- **Precision > 80%** to avoid customer annoyance
- **F1-Score > 77%** for balanced performance
- **Business Impact:** $2.3M potential revenue saved annually

**📊 Dataset Source:** 
- **Telco Customer Churn Dataset** (7,043 customers)
- **Features:** Demographics, services, account info, charges
- **Target:** Binary classification (Churn: Yes/No)
- **Class Distribution:** 73.5% retained, 26.5% churned
- [Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)


**🔧 Technical Highlights:**
- The raw data contains **7043 rows** (customers) and **21 columns** (features).
- Advanced feature engineering with behavioral patterns
- Handling class imbalance using SMOTE and cost-sensitive learning
- Ensemble methods (Random Forest + XGBoost + Logistic Regression)
- Hyperparameter optimization with Bayesian search
- Customer segmentation analysis
- Threshold optimization for business metrics

**🛠️ Tech Stack:** Python, Pandas, Scikit-learn, XGBoost, SMOTE, Streamlit, Plotly

**📸 Output Preview:**
![Churn Prediction Results](https://github.com/Sumit-Kumar-Dash/Top-5-Data-Science-Projects/blob/main/customer_churn_prediction_dashboard.png)

---

### 2. GenAI Document Intelligence System
> **AI-powered document analysis and question-answering system using Large Language Models**

**📄 Business Problem:** Organizations struggle to extract insights from large volumes of unstructured documents (contracts, reports, manuals). Employees spend 40% of their time searching for information, costing companies millions in productivity loss.

**🎯 Target Metrics:**
- **90%+ accuracy** in document Q&A tasks
- **75% time reduction** in information retrieval
- **Support for 10+ file formats** (PDF, Word, Images, etc.)
- **Sub-5 second response time** for complex queries

**📊 Dataset Source:** 
- **Custom enterprise document collection** (1000+ documents)
- **Public datasets:** Squad 2.0, MS-MARCO for benchmarking
- **Document types:** Technical manuals, legal contracts, research papers
- **Languages:** English, with multilingual support

**🔧 Technical Highlights:**
- Retrieval-Augmented Generation (RAG) architecture
- Vector embeddings with OpenAI/Sentence-Transformers
- Document chunking with semantic awareness
- Multi-modal processing (text + images with OCR)
- Conversation memory and context management
- Similarity search with ChromaDB/Pinecone
- Response evaluation and citation tracking

**🛠️ Tech Stack:** Python, LangChain, OpenAI API, ChromaDB, FastAPI, Streamlit

**📸 Output Preview:**
![Document Q&A Interface](https://github.com/Sumit-Kumar-Dash/Top-5-Data-Science-Projects/blob/main/chatbot.png)

---

### 3. Time Series Forecasting Pipeline
> **Multi-horizon demand forecasting with automated model selection and deployment**

![Time Series Dashboard](https://github.com/Sumit-Kumar-Dash/Top-5-Data-Science-Projects/blob/main/retail_demand_dashboard.png)

**📈 Business Problem:** Retail companies struggle with inventory management due to inaccurate demand forecasting. Poor forecasting leads to stockouts (lost sales) or overstock (carrying costs), resulting in millions in losses annually.

**🎯 Target Metrics:**
- **MAPE < 10%** for short-term forecasts (1-4 weeks)
- **RMSE improvement > 15%** over baseline methods
- **95% prediction intervals** for uncertainty quantification
- **Business Impact:** $500K annual savings in inventory costs

**📊 Dataset Source:** 
- **Retail Store Demand Forecast** (76000 rows, 2 years)
- **Forecast Target**: Demand (units sold per day)
- **Features:** Historical demand, promotion flags, discount rates, inventory levels
- **External data:** Weather, weather condition, holiday indicators
- **Engineered features:** lag values, rolling averages, standard deviations
- [Dataset](https://www.kaggle.com/datasets/atomicd/retail-store-inventory-and-demand-forecasting)

**🔧 Technical Highlights:**
- Classical time series methods (ARIMA, Exponential Smoothing)
- Modern ML approaches (XGBoost Regressor, LSTM, Transformer models)
- Feature engineering with lag variables and rolling statistics
- Automated hyperparameter optimization with Optuna
- Time-aware splitting: Avoided data leakage using chronological train–validation strategy
- Seasonal decomposition and trend analysis

**🛠️ Tech Stack:** Python, Prophet, TensorFlow, Optuna, MLflow, Streamlit, Plotly

**📸 Output Preview:**
![Forecasting Dashboard](https://github.com/Sumit-Kumar-Dash/Top-5-Data-Science-Projects/blob/main/time_series_demand_forecast_dashboard.png)

---

### 4. E-commerce Product Reviews Scraping & Analysis
> **Automated web scraping and sentiment analysis of product reviews for competitive intelligence**

**🛒 Business Problem:** E-commerce companies need to monitor product performance and customer feedback across multiple platforms. Manual review monitoring is time-consuming and doesn't scale. Companies lose competitive advantage without insights into customer sentiment and product issues.

**🎯 Target Metrics:**
- **Scrape 10,000+ reviews** daily across multiple platforms
- **90%+ accuracy** in review classification (positive, neutral, negative)
- **<2 seconds/review NLP processing time**
- **Real-time competitive analysis** across 5+ competitors
- **Business Value**: Drives competitive pricing and product strategy — estimated 15% conversion improvement

**📊 Dataset Source:** 
- **Amazon Product Reviews** (scraped via BeautifulSoup/Selenium)
- **Flipkart, eBay Reviews** (multi-platform scraping)
- **Target Products:** Electronics, Fashion, Home & Garden
- **Review Fields**: Title, Body, Rating, Date, Verified Badge, Review Votes
- [Dataset](https://www.kaggle.com/datasets/asaniczka/amazon-brazil-products-2023-1-3m-products/code)

**🔧 Technical Highlights:**
- Data Cleaning NLP preprocessing: Spell correction, lemmatization, emoji filtering
- Multi-platform web scraping
- Distributed scraping with Scrapy framework
- Sentiment classification
- Aspect-based sentiment analysis (price, quality, delivery)
- Review summarization using extractive methods
- Topic modeling (LDA, BERTopic) to extract customer pain points

**🛠️ Tech Stack:** Python, Selenium, BeautifulSoup, Pandas, NLTK, MongoDB, TextBlob, Transformers, LDA

---

### 5. Healthcare Analytics: Disease Risk Prediction
> **Comprehensive data analysis and machine learning for early disease detection**

**🏥 Business Problem:** Healthcare systems struggle with early disease detection and resource allocation. Late diagnosis leads to higher treatment costs and worse patient outcomes. Hospitals need predictive models to identify high-risk patients and optimize preventive care strategies. Aims to leverage data science and machine learning for early detection and prevention of chronic diseases.

**🎯 Target Metrics:**
- **Sensitivity > 90%** for disease detection (minimize false negatives)
- **Specificity > 85%** to reduce unnecessary treatments
- **AUC-ROC > 0.92** for overall model performance
- **Cost savings:** $2M annually through early intervention

**📊 Dataset Source:** 
- **Heart Disease ViT Dataset** 70,000 patient data
- **Binary Classification**  having Balanced data predict risk_label of heart disease
- **Custom synthetic data** generated for privacy compliance
- **Features:** Symptoms (Binary - Yes/No) and Risk Factors (Binary - Yes/No or Continuous)
- [Dataset](https://www.kaggle.com/datasets/mahatiratusher/heart-disease-risk-prediction-dataset/data)

**🔧 Technical Highlights:**
- Extensive exploratory data analysis with statistical tests
- Statistical analysis: correlation matrices, hypothesis testing, heatmap
- Multiple ML algorithms comparison (Logistic Regression, Random Forest, XGBoost)
- Feature importance analysis 
- Clinical decision threshold optimization
- Comprehensive data visualization with medical insights

**🛠️ Tech Stack:** Python, Pandas, NumPy, Scikit-learn, Seaborn, Plotly, XGBoost, Streamlit

---

## 🤝 Contributing

I welcome contributions, suggestions, and feedback! Here's how you can help:

1. **🐛 Report Issues**: Found a bug? [Open an issue](https://github.com/Sumit-Kumar-Dash/Top-5-Data-Science-Projects/issues)
2. **💡 Suggest Features**: Have ideas for improvements? Let's discuss!
3. **🔧 Submit PRs**: Code improvements are always welcome
4. **⭐ Star the Repo**: If you find this helpful, please star it!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE] file for details.

---

<div align="center">

### ⭐ If this repository helped you land your first data science job, please star it and share your success story!

**[📺 Watch the Full YouTube Tutorial](https://youtu.be/IaxTPdJoy8o)**

</div>

---

<div align="center">
<sub>Built with ❤️ by [Sumit Kumar Dash] | Last Updated: June 2025</sub>
</div>





